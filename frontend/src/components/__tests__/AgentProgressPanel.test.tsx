/**
 * AgentProgressPanel Tests
 * 
 * Comprehensive test suite for AgentProgressPanel component.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import AgentProgressPanel from '../AgentProgressPanel';
import { AgentProgress } from '../../hooks/useAgentStream';

describe('AgentProgressPanel', () => {
  const mockProgress: AgentProgress = {
    task_id: 'test-task-123',
    agent_id: 'distillation_v2',
    status: 'running',
    progress_pct: 62.5,
    time_elapsed_sec: 125,
    estimated_time_remaining_sec: 75,
    current_step: 'Generating executive summary with LLM...',
    total_steps: 8,
    completed_steps: 5,
  };

  it('renders null state when no progress', () => {
    render(<AgentProgressPanel progress={null} />);
    expect(screen.getByText(/waiting for analysis to start/i)).toBeInTheDocument();
  });

  it('renders progress data correctly', () => {
    render(<AgentProgressPanel progress={mockProgress} />);
    
    // Check status text
    expect(screen.getByText('Analysis In Progress')).toBeInTheDocument();
    
    // Check agent ID
    expect(screen.getByText('distillation_v2')).toBeInTheDocument();
    
    // Check progress percentage
    expect(screen.getByText('62.5%')).toBeInTheDocument();
    
    // Check step count
    expect(screen.getByText('Step 5 of 8')).toBeInTheDocument();
    
    // Check current step
    expect(screen.getByText('Generating executive summary with LLM...')).toBeInTheDocument();
  });

  it('formats time correctly', () => {
    render(<AgentProgressPanel progress={mockProgress} />);
    
    // 125 seconds = 2:05
    expect(screen.getByText('2:05')).toBeInTheDocument();
    
    // 75 seconds = 1:15
    expect(screen.getByText('1:15')).toBeInTheDocument();
  });

  it('shows completed status', () => {
    const completedProgress: AgentProgress = {
      ...mockProgress,
      status: 'completed',
      progress_pct: 100,
      completed_steps: 8,
    };
    
    render(<AgentProgressPanel progress={completedProgress} />);
    expect(screen.getByText('Analysis Complete')).toBeInTheDocument();
  });

  it('shows failed status', () => {
    const failedProgress: AgentProgress = {
      ...mockProgress,
      status: 'failed',
    };
    
    render(<AgentProgressPanel progress={failedProgress} />);
    expect(screen.getByText('Analysis Failed')).toBeInTheDocument();
  });

  it('shows pending status', () => {
    const pendingProgress: AgentProgress = {
      ...mockProgress,
      status: 'pending',
      progress_pct: 0,
      completed_steps: 0,
    };
    
    render(<AgentProgressPanel progress={pendingProgress} />);
    expect(screen.getByText('Analysis Pending')).toBeInTheDocument();
  });

  it('handles missing estimated time remaining', () => {
    const progressWithoutEstimate: AgentProgress = {
      ...mockProgress,
      estimated_time_remaining_sec: undefined,
    };
    
    render(<AgentProgressPanel progress={progressWithoutEstimate} />);
    expect(screen.getByText('--:--')).toBeInTheDocument();
  });

  it('clamps progress percentage to 0-100', () => {
    const overProgress: AgentProgress = {
      ...mockProgress,
      progress_pct: 150,
    };
    
    render(<AgentProgressPanel progress={overProgress} />);
    expect(screen.getByText('100.0%')).toBeInTheDocument();
  });

  it('displays task ID', () => {
    render(<AgentProgressPanel progress={mockProgress} />);
    expect(screen.getByText(/test-task-123/)).toBeInTheDocument();
  });

  it('shows estimated completion time for running status', () => {
    render(<AgentProgressPanel progress={mockProgress} />);
    expect(screen.getByText('Estimated Completion')).toBeInTheDocument();
  });

  it('hides estimated completion time for completed status', () => {
    const completedProgress: AgentProgress = {
      ...mockProgress,
      status: 'completed',
    };
    
    render(<AgentProgressPanel progress={completedProgress} />);
    expect(screen.queryByText('Estimated Completion')).not.toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(<AgentProgressPanel progress={mockProgress} className="custom-class" />);
    expect(container.querySelector('.custom-class')).toBeInTheDocument();
  });

  it('renders progress bar with correct width', () => {
    const { container } = render(<AgentProgressPanel progress={mockProgress} />);
    const progressBar = container.querySelector('[style*="width: 62.5%"]');
    expect(progressBar).toBeInTheDocument();
  });

  it('shows loading spinner for running status', () => {
    const { container } = render(<AgentProgressPanel progress={mockProgress} />);
    const spinner = container.querySelector('.animate-spin');
    expect(spinner).toBeInTheDocument();
  });

  it('shows check icon for completed status', () => {
    const completedProgress: AgentProgress = {
      ...mockProgress,
      status: 'completed',
    };
    
    const { container } = render(<AgentProgressPanel progress={completedProgress} />);
    // CheckCircle2 icon should be present
    expect(container.querySelector('svg')).toBeInTheDocument();
  });

  it('shows error icon for failed status', () => {
    const failedProgress: AgentProgress = {
      ...mockProgress,
      status: 'failed',
    };
    
    const { container } = render(<AgentProgressPanel progress={failedProgress} />);
    // XCircle icon should be present
    expect(container.querySelector('svg')).toBeInTheDocument();
  });
});

