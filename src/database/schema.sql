-- Options Analysis System Database Schema
-- PostgreSQL 15+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- User preferences
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    risk_tolerance VARCHAR(20) DEFAULT 'moderate', -- conservative, moderate, aggressive
    notification_preferences JSONB DEFAULT '{}',
    report_frequency VARCHAR(20) DEFAULT 'daily', -- daily, weekly, on_demand
    preferred_strategies JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Positions table
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    strategy_type VARCHAR(50), -- long_call, iron_condor, etc.
    entry_date DATE NOT NULL,
    expiration_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'open', -- open, closed, expired
    total_premium DECIMAL(12, 2),
    market_value DECIMAL(12, 2),
    pnl DECIMAL(12, 2),
    pnl_pct DECIMAL(8, 4),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_positions (user_id, status),
    INDEX idx_symbol (symbol),
    INDEX idx_expiration (expiration_date)
);

-- Position legs (individual options in a position)
CREATE TABLE position_legs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID REFERENCES positions(id) ON DELETE CASCADE,
    option_type VARCHAR(10) NOT NULL, -- call, put
    strike DECIMAL(10, 2) NOT NULL,
    quantity INTEGER NOT NULL,
    is_short BOOLEAN DEFAULT FALSE,
    entry_price DECIMAL(10, 4) NOT NULL,
    current_price DECIMAL(10, 4),
    multiplier INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_position_legs (position_id)
);

-- Market data cache
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    underlying_price DECIMAL(10, 2),
    iv DECIMAL(8, 4), -- Implied volatility
    historical_iv DECIMAL(8, 4),
    iv_rank DECIMAL(5, 2),
    volume BIGINT,
    avg_volume BIGINT,
    put_call_ratio DECIMAL(8, 4),
    days_to_earnings INTEGER,
    sector VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_timestamp (timestamp)
);

-- Greeks calculations
CREATE TABLE greeks_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID REFERENCES positions(id) ON DELETE CASCADE,
    delta DECIMAL(12, 4),
    gamma DECIMAL(12, 6),
    theta DECIMAL(12, 4),
    vega DECIMAL(12, 4),
    rho DECIMAL(12, 4),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_position_greeks (position_id, calculated_at)
);

-- Expected value calculations
CREATE TABLE ev_calculations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID REFERENCES positions(id) ON DELETE CASCADE,
    expected_value DECIMAL(12, 2),
    expected_return_pct DECIMAL(8, 4),
    probability_profit DECIMAL(5, 4),
    confidence_interval_lower DECIMAL(12, 2),
    confidence_interval_upper DECIMAL(12, 2),
    method_breakdown JSONB,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_position_ev (position_id, calculated_at)
);

-- Analysis reports
CREATE TABLE analysis_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    report_type VARCHAR(50) NOT NULL, -- daily, weekly, pre_market, etc.
    executive_summary TEXT,
    market_overview TEXT,
    portfolio_analysis TEXT,
    risk_assessment TEXT,
    recommendations JSONB,
    action_items JSONB,
    risk_score DECIMAL(5, 2),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_reports (user_id, generated_at),
    INDEX idx_report_type (report_type, generated_at)
);

-- Risk alerts
CREATE TABLE risk_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    position_id UUID REFERENCES positions(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
    message TEXT NOT NULL,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    INDEX idx_user_alerts (user_id, is_acknowledged, created_at),
    INDEX idx_severity (severity, created_at)
);

-- Agent execution logs
CREATE TABLE agent_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    execution_type VARCHAR(50), -- scheduled, on_demand
    status VARCHAR(20) NOT NULL, -- running, completed, failed
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_ms INTEGER,
    error_message TEXT,
    metadata JSONB,
    INDEX idx_agent_executions (agent_name, start_time),
    INDEX idx_status (status, start_time)
);

-- Trades history (for closed positions)
CREATE TABLE trades_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    position_id UUID REFERENCES positions(id) ON DELETE SET NULL,
    symbol VARCHAR(20) NOT NULL,
    strategy_type VARCHAR(50),
    entry_date DATE NOT NULL,
    exit_date DATE NOT NULL,
    days_held INTEGER,
    entry_premium DECIMAL(12, 2),
    exit_premium DECIMAL(12, 2),
    realized_pnl DECIMAL(12, 2),
    realized_pnl_pct DECIMAL(8, 4),
    max_profit DECIMAL(12, 2),
    max_loss DECIMAL(12, 2),
    exit_reason VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_trades (user_id, exit_date),
    INDEX idx_symbol_trades (symbol, exit_date)
);

-- Scheduled tasks
CREATE TABLE scheduled_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_name VARCHAR(100) NOT NULL,
    schedule_type VARCHAR(50) NOT NULL, -- pre_market, market_open, mid_day, end_of_day
    schedule_time TIME NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMP,
    next_run TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_scheduled_tasks (is_active, next_run)
);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_position_legs_updated_at BEFORE UPDATE ON position_legs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries

-- Active positions with current Greeks
CREATE VIEW v_active_positions_with_greeks AS
SELECT 
    p.*,
    g.delta,
    g.gamma,
    g.theta,
    g.vega,
    g.rho,
    g.calculated_at as greeks_timestamp
FROM positions p
LEFT JOIN LATERAL (
    SELECT * FROM greeks_history
    WHERE position_id = p.id
    ORDER BY calculated_at DESC
    LIMIT 1
) g ON TRUE
WHERE p.status = 'open';

-- Portfolio summary by user
CREATE VIEW v_portfolio_summary AS
SELECT 
    user_id,
    COUNT(*) as total_positions,
    SUM(market_value) as total_market_value,
    SUM(pnl) as total_pnl,
    AVG(pnl_pct) as avg_pnl_pct,
    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_positions,
    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_positions
FROM positions
WHERE status = 'open'
GROUP BY user_id;

-- Recent alerts by user
CREATE VIEW v_recent_alerts AS
SELECT 
    ra.*,
    p.symbol,
    p.strategy_type
FROM risk_alerts ra
LEFT JOIN positions p ON ra.position_id = p.id
WHERE ra.is_acknowledged = FALSE
ORDER BY ra.created_at DESC;

