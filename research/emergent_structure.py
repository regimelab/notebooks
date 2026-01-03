import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class Agent:
    """Heterogeneous agent with sensitive entry/exit rules and psychology."""
    def __init__(self, id, initial_wealth=1.0, risk_aversion=1.0, leverage=1.0, 
                 stop_loss=-0.2, target_profit=0.3, vol_sensitivity=1.0):
        self.id = id
        self.w = initial_wealth  # Current wealth
        self.w0 = initial_wealth  # Initial wealth
        self.risk_aversion = risk_aversion  # Affects position sizing
        self.leverage = leverage
        self.stop_loss = stop_loss  # Psychological stop
        self.target_profit = target_profit
        self.vol_sensitivity = vol_sensitivity  # How vol affects psychology
        
        # Hidden psychological state: confidence (1=normal, <0.5=stress, >1.2=euphoria)
        self.z = 1.0
        
        # Position tracking
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.max_unrealized_pnl = 0.0  # For drawdown calc
        
    def update_psychology(self, recent_vol, recent_return):
        """Update hidden state based on vol and returns."""
        stress = -recent_return * self.vol_sensitivity * recent_vol
        self.z = 0.95 * self.z + 0.05 * np.exp(stress)  # EWMA update
        self.z = np.clip(self.z, 0.1, 2.0)
        
    def decide_position(self, price, recent_vol):
        """Sensitive entry/exit based on thresholds and psychology."""
        unrealized_return = (price - self.entry_price) / self.entry_price if self.position != 0 else 0
        
        # Update max drawdown tracking
        if self.position > 0:
            self.max_unrealized_pnl = max(self.max_unrealized_pnl, unrealized_return)
            drawdown = unrealized_return - self.max_unrealized_pnl
        
        # Exit conditions: stop loss, target, or psychological
        if (self.position > 0 and 
            (unrealized_return < self.stop_loss or 
             unrealized_return > self.target_profit or
             (self.z < 0.6 and recent_vol > 0.05))):  # Panic exit
            self.position = 0
            
        # Entry: only if not positioned and psychology allows
        elif self.position == 0 and self.z > 0.8:
            # Position size sensitive to risk aversion, leverage, vol
            pos_size = (1.0 / (1 + self.risk_aversion * recent_vol)) * self.leverage * self.z
            self.position = pos_size
            self.entry_price = price
            self.max_unrealized_pnl = 0.0
            
        return unrealized_return if self.position != 0 else 0

class CrashSimulator:
    """Simulates wealth processes with structural crashes from crowding."""
    def __init__(self, n_agents=1000, n_steps=1000, dt=1/252):
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.dt = dt
        
        # Heterogeneous agents
        rav = np.random.lognormal(0, 0.5, n_agents)  # Risk aversion
        lev = np.random.uniform(0.5, 3.0, n_agents)  # Leverage
        sl = np.random.uniform(-0.4, -0.05, n_agents)  # Stop losses
        tp = np.random.uniform(0.1, 0.8, n_agents)  # Targets
        vs = np.random.uniform(0.5, 2.0, n_agents)  # Vol sensitivity
        
        self.agents = [Agent(i, risk_aversion=rav[i], leverage=lev[i], 
                            stop_loss=sl[i], target_profit=tp[i], 
                            vol_sensitivity=vs[i]) for i in range(n_agents)]
        
        # Price process state (multifractal volatility cascade style)
        self.price = 100.0
        self.log_price = np.log(self.price)
        self.recent_vol = 0.02  # EWMA vol
        self.lambda_vol = 0.94  # Vol persistence
        
    def multifractal_vol(self):
        """Generate clustered volatility (Mandelbrot-style)."""
        # Long-memory vol shock
        hurst = 0.8  # Persistent
        vol_shock = (np.random.standard_normal() * 
                    (self.recent_vol ** 2) ** 0.5 * 
                    self.recent_vol ** (2 * hurst - 1))
        self.recent_vol = np.sqrt(self.lambda_vol * self.recent_vol**2 + 
                                 (1 - self.lambda_vol) * vol_shock)
        return self.recent_vol * np.sqrt(self.dt)
    
    def step(self):
        """Single time step: price move, agent updates, crowding check."""
        # Generate price change with multifractal vol (heavy tails)
        vol = self.multifractal_vol()
        epsilon = stats.t.rvs(df=3)  # Fat tails
        dp = 0.0002 * self.dt + vol * epsilon  # Small drift
        
        # Market impact from crowding (agent synchronization)
        active_agents = sum(1 for a in self.agents if a.position != 0)
        crowding = min(active_agents / self.n_agents, 1.0)
        impact = -0.5 * crowding * vol * np.sign(dp)  # Herding amplifies
        
        self.log_price += dp + impact
        self.price = np.exp(self.log_price)
        
        # Agent updates
        recent_return = dp / self.dt if abs(dp) > 1e-8 else 0
        for agent in self.agents:
            agent.update_psychology(self.recent_vol, recent_return)
            unreal_return = agent.decide_position(self.price, self.recent_vol)
            
            # Wealth update
            if agent.position != 0:
                agent.w *= (1 + self.leverage * unreal_return * self.dt / self.dt)
        
        return crowding
    
    def run_simulation(self):
        """Run full simulation."""
        crowd_hist = np.zeros(self.n_steps)
        wealths = np.zeros((self.n_steps, self.n_agents))
        
        for t in range(self.n_steps):
            crowding = self.step()
            crowd_hist[t] = crowding
            
            for i, agent in enumerate(self.agents):
                wealths[t, i] = agent.w
        
        return crowd_hist, wealths, self.price
    
    def identify_structures(self, crowd_hist, wealths, prices):
        """Find 'crash structures': high crowding + vol regions."""
        returns = np.diff(np.log(prices))
        vols = pd.Series(returns).rolling(10).std() * np.sqrt(252)
        
        high_crowd = crowd_hist > np.percentile(crowd_hist, 80)
        high_vol = vols > np.percentile(vols, 80)
        structures = high_crowd & high_vol
        
        return structures

# Run simulation
sim = CrashSimulator(n_agents=2000, n_steps=2000)
crowd_hist, wealths, price_path = sim.run_simulation()

# Analysis
structures = sim.identify_structures(crowd_hist, wealths, np.exp(np.cumsum(np.diff(np.log(sim.price)) if len(sim.price)>1 else np.log(sim.price))))

print("Simulation complete.")
print(f"Crash structures detected: {structures.sum()} out of {len(structures)} periods [web:11]")

# Correlation against structure
normal_wealth_corr = np.corrcoef(wealths[structures==0].T)[-1, -2] if structures.sum() < len(structures)-1 else 0
struct_wealth_corr = np.corrcoef(wealths[structures].T)[-1, -2] if structures.sum() > 1 else 0

print(f"Normal regime agent wealth correlation: {normal_wealth_corr:.3f}")
print(f"Crash structure correlation: {struct_wealth_corr:.3f} [web:14]")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price and crowding
axes[0,0].plot(price_path)
axes[0,0].set_title('Price Path (Structural Crashes)')

axes[0,1].plot(crowd_hist)
axes[0,1].set_title('Crowding (Synchronization)')

# Wealth distribution over time
sns.heatmap(wealths[::10].T, ax=axes[1,0], cmap='viridis')
axes[1,0].set_title('Agent Wealth Heatmap (Rows=Agents)')

# Correlations by regime
corr_normal = [np.corrcoef(wealths[t])[0,1] for t in range(1, len(wealths)-1) if not structures[t]]
corr_struct = [np.corrcoef(wealths[t])[0,1] for t in range(1, len(wealths)-1) if structures[t]]

axes[1,1].hist(corr_normal, alpha=0.5, label='Normal', bins=30)
axes[1,1].hist(corr_struct, alpha=0.5, label='Structure', bins=30)
axes[1,1].legend()
axes[1,1].set_title('Cross-Agent Wealth Correlation by Regime')
axes[1,1].axvline(0, color='k', linestyle='--')

plt.tight_layout()
plt.show()
      
