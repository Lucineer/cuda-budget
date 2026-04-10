/*!
# cuda-budget

Resource budgeting and spending limits.

Agents need budgets — computation time, API tokens, memory, network.
This crate tracks allocation, spending, and enforcement with
priority-based scheduling.

- Budget with multiple resource types
- Allocation and spending tracking
- Priority-based scheduling
- Spending limits and alerts
- Budget transfer between agents
- Forecasting (spend rate projection)
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Resource types an agent might spend
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    ComputeMs,      // CPU/GPU time in ms
    Tokens,         // LLM tokens
    MemoryBytes,    // RAM usage
    NetworkBytes,   // bandwidth
    MoneyCents,     // monetary cost
}

/// Priority level
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority { Low = 0, Normal = 1, High = 2, Critical = 3 }

impl ResourceType {
    pub fn unit(&self) -> &'static str {
        match self { ResourceType::ComputeMs => "ms", ResourceType::Tokens => "tok", ResourceType::MemoryBytes => "B", ResourceType::NetworkBytes => "B", ResourceType::MoneyCents => "¢" }
    }
}

/// A budget allocation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Allocation {
    pub resource: ResourceType,
    pub allocated: f64,
    pub spent: f64,
    pub limit: f64,
    pub alert_threshold: f64, // fraction 0.0-1.0
}

impl Allocation {
    pub fn remaining(&self) -> f64 { (self.allocated - self.spent).max(0.0) }
    pub fn utilization(&self) -> f64 { if self.allocated == 0.0 { 1.0 } else { self.spent / self.allocated } }
    pub fn is_exhausted(&self) -> bool { self.spent >= self.allocated }
    pub fn is_over_limit(&self) -> bool { self.spent >= self.limit }
    pub fn should_alert(&self) -> bool { self.utilization() >= self.alert_threshold }
}

/// An agent budget
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentBudget {
    pub agent_id: String,
    pub allocations: HashMap<ResourceType, Allocation>,
    pub priority: Priority,
    pub created: u64,
    pub total_alerts: u64,
}

impl AgentBudget {
    pub fn new(agent_id: &str, priority: Priority) -> Self { AgentBudget { agent_id: agent_id.to_string(), allocations: HashMap::new(), priority, created: now(), total_alerts: 0 } }

    /// Allocate a resource
    pub fn allocate(&mut self, resource: ResourceType, amount: f64, limit: f64) {
        self.allocations.insert(resource, Allocation { resource, allocated: amount, spent: 0.0, limit, alert_threshold: 0.8 });
    }

    /// Try to spend from a budget. Returns true if allowed.
    pub fn spend(&mut self, resource: ResourceType, amount: f64) -> SpendResult {
        let alloc = match self.allocations.get_mut(&resource) {
            Some(a) => a,
            None => return SpendResult::NoAllocation,
        };
        if alloc.is_over_limit() { return SpendResult::OverLimit; }
        if alloc.is_exhausted() { return SpendResult::Exhausted; }
        alloc.spent += amount;
        if alloc.should_alert() { self.total_alerts += 1; SpendResult::Alert }
        else { SpendResult::Ok }
    }

    /// Transfer budget between agents
    pub fn transfer_to(&mut self, resource: ResourceType, amount: f64, other: &mut AgentBudget) -> TransferResult {
        let my_alloc = match self.allocations.get_mut(&resource) { Some(a) => a, None => return TransferResult::NoAllocation };
        let their_alloc = match other.allocations.get_mut(&resource) { Some(a) => a, None => return TransferResult::NoAllocation };
        if my_alloc.remaining() < amount { return TransferResult::Insufficient; }
        my_alloc.spent += amount;
        their_alloc.allocated += amount;
        TransferResult::Ok
    }

    /// Forecast: time until exhaustion at current rate
    pub fn forecast(&self, resource: ResourceType, elapsed_ms: f64) -> f64 {
        let alloc = match self.allocations.get(&resource) { Some(a) => a, None => return f64::INFINITY };
        if alloc.spent == 0.0 || elapsed_ms <= 0.0 { return f64::INFINITY; }
        let rate = alloc.spent / (elapsed_ms / 1000.0); // per second
        let remaining = alloc.remaining();
        if rate <= 0.0 { return f64::INFINITY; }
        remaining / rate // seconds until exhaustion
    }

    /// Get all active alerts
    pub fn alerts(&self) -> Vec<ResourceType> {
        self.allocations.values().filter(|a| a.should_alert()).map(|a| a.resource).collect()
    }

    pub fn summary(&self) -> String {
        format!("Budget[{}]: priority={:?}, {} resources, alerts={}",
            self.agent_id, self.priority, self.allocations.len(), self.total_alerts)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpendResult { Ok, Alert, Exhausted, OverLimit, NoAllocation }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferResult { Ok, NoAllocation, Insufficient }

/// Fleet budget manager
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetManager {
    pub agents: HashMap<String, AgentBudget>,
    pub global_limits: HashMap<ResourceType, f64>,
    pub total_spent: HashMap<ResourceType, f64>,
}

impl BudgetManager {
    pub fn new() -> Self { BudgetManager { agents: HashMap::new(), global_limits: HashMap::new(), total_spent: HashMap::new() } }

    pub fn create_budget(&mut self, agent_id: &str, priority: Priority) {
        self.agents.insert(agent_id.to_string(), AgentBudget::new(agent_id, priority));
    }

    pub fn allocate(&mut self, agent_id: &str, resource: ResourceType, amount: f64, limit: f64) {
        if let Some(budget) = self.agents.get_mut(agent_id) { budget.allocate(resource, amount, limit); }
    }

    pub fn spend(&mut self, agent_id: &str, resource: ResourceType, amount: f64) -> SpendResult {
        if let Some(budget) = self.agents.get_mut(agent_id) {
            let result = budget.spend(resource, amount);
            *self.total_spent.entry(resource).or_insert(0.0) += amount;
            return result;
        }
        SpendResult::NoAllocation
    }

    /// Priority queue: critical first, then by remaining budget
    pub fn priority_queue(&self) -> Vec<&AgentBudget> {
        let mut agents: Vec<_> = self.agents.values().collect();
        agents.sort_by(|a, b| {
            match b.priority.cmp(&a.priority) {
                std::cmp::Ordering::Equal => {
                    let avg_a: f64 = a.allocations.values().map(|al| al.utilization()).sum::<f64>() / a.allocations.len().max(1) as f64;
                    let avg_b: f64 = b.allocations.values().map(|al| al.utilization()).sum::<f64>() / b.allocations.len().max(1) as f64;
                    avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
                }
                ord => ord,
            }
        });
        agents
    }

    pub fn summary(&self) -> String {
        let total: f64 = self.total_spent.values().sum();
        format!("BudgetManager: {} agents, total_spent={:.1}", self.agents.len(), total)
    }
}

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_spend() {
        let mut budget = AgentBudget::new("a1", Priority::Normal);
        budget.allocate(ResourceType::Tokens, 1000.0, 1500.0);
        assert_eq!(budget.spend(ResourceType::Tokens, 500.0), SpendResult::Ok);
        assert_eq!(budget.allocations[&ResourceType::Tokens].remaining(), 500.0);
    }

    #[test]
    fn test_exhausted() {
        let mut budget = AgentBudget::new("a1", Priority::Normal);
        budget.allocate(ResourceType::Tokens, 100.0, 200.0);
        budget.spend(ResourceType::Tokens, 100.0);
        assert_eq!(budget.spend(ResourceType::Tokens, 1.0), SpendResult::Exhausted);
    }

    #[test]
    fn test_over_limit() {
        let mut budget = AgentBudget::new("a1", Priority::Normal);
        budget.allocate(ResourceType::Tokens, 1000.0, 1000.0);
        budget.spend(ResourceType::Tokens, 1000.0);
        assert_eq!(budget.spend(ResourceType::Tokens, 1.0), SpendResult::OverLimit);
    }

    #[test]
    fn test_alert_threshold() {
        let mut budget = AgentBudget::new("a1", Priority::Normal);
        budget.allocate(ResourceType::Tokens, 100.0, 200.0);
        budget.spend(ResourceType::Tokens, 81.0); // > 80%
        let alerts = budget.alerts();
        assert!(alerts.contains(&ResourceType::Tokens));
    }

    #[test]
    fn test_transfer() {
        let mut a = AgentBudget::new("a1", Priority::Normal);
        let mut b = AgentBudget::new("a2", Priority::Normal);
        a.allocate(ResourceType::Tokens, 1000.0, 2000.0);
        b.allocate(ResourceType::Tokens, 0.0, 2000.0);
        assert_eq!(a.transfer_to(ResourceType::Tokens, 200.0, &mut b), TransferResult::Ok);
        assert_eq!(b.allocations[&ResourceType::Tokens].allocated, 200.0);
    }

    #[test]
    fn test_forecast() {
        let mut budget = AgentBudget::new("a1", Priority::Normal);
        budget.allocate(ResourceType::Tokens, 1000.0, 2000.0);
        budget.spend(ResourceType::Tokens, 100.0); // spent 100 in some time
        let remaining_sec = budget.forecast(ResourceType::Tokens, 10000.0); // 10 seconds elapsed
        assert!(remaining_sec.is_finite());
        assert!(remaining_sec > 0.0);
    }

    #[test]
    fn test_priority_queue() {
        let mut mgr = BudgetManager::new();
        mgr.create_budget("critical", Priority::Critical);
        mgr.create_budget("normal", Priority::Normal);
        mgr.create_budget("low", Priority::Low);
        let queue = mgr.priority_queue();
        assert_eq!(queue[0].agent_id, "critical");
    }

    #[test]
    fn test_budget_manager_summary() {
        let mgr = BudgetManager::new();
        let s = mgr.summary();
        assert!(s.contains("0 agents"));
    }

    #[test]
    fn test_no_allocation_spend() {
        let mut budget = AgentBudget::new("a1", Priority::Normal);
        assert_eq!(budget.spend(ResourceType::Tokens, 1.0), SpendResult::NoAllocation);
    }
}
