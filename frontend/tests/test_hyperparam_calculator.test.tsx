/**
 * Frontend tests for hyperparameter calculation UI components.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HyperparamCalculator } from '@/components/hyperparam-calculator';
import { api } from '@/lib/api';

// Mock the API
vi.mock('@/lib/api');

const mockApi = api as any;

describe('HyperparamCalculator', () => {
  beforeEach(() => {
    // Reset mocks
    vi.clearAllMocks();

    // Mock successful API response
    mockApi.calculateHyperparameters = vi.fn().mockResolvedValue({
      success: true,
      model_name: 'meta-llama/Llama-3.1-8B',
      recipe_type: 'sft',
      recommendations: {
        learning_rate: 2.86e-04,
        batch_size: 64,
        lora_rank: 64,
        adam_beta1: 0.9,
        adam_beta2: 0.95,
        adam_eps: 1e-8
      },
      explanation: {
        learning_rate: 'LR = 5e-5 × 10 × (2000/4096)^0.781 = 2.86e-04',
        batch_size: 'Optimized for sft training. Tinker docs recommend 128 or smaller for SFT.',
        lora_rank: 'Default is 32 for most use cases. Independent of learning rate.',
        notes: [
          'Learning rate is independent of LoRA rank',
          'Batch size: 128 or smaller recommended for SFT',
          'LoRA requires ~10x higher LR than full fine-tuning'
        ],
        source: 'https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams'
      }
    });
  });

  it('renders hyperparameter calculator form', () => {
    render(<HyperparamCalculator />);

    expect(screen.getByText('Hyperparameter Calculator')).toBeInTheDocument();
    expect(screen.getByLabelText('Model')).toBeInTheDocument();
    expect(screen.getByLabelText('Recipe Type')).toBeInTheDocument();
    expect(screen.getByLabelText('LoRA Rank (optional)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Calculate' })).toBeInTheDocument();
  });

  it('calculates hyperparameters successfully', async () => {
    render(<HyperparamCalculator />);

    // Fill form
    fireEvent.change(screen.getByLabelText('Model'), {
      target: { value: 'meta-llama/Llama-3.1-8B' }
    });
    fireEvent.change(screen.getByLabelText('Recipe Type'), {
      target: { value: 'sft' }
    });
    fireEvent.change(screen.getByLabelText('LoRA Rank (optional)'), {
      target: { value: '64' }
    });

    // Submit form
    fireEvent.click(screen.getByRole('button', { name: 'Calculate' }));

    // Wait for results
    await waitFor(() => {
      expect(screen.getByText('Learning Rate')).toBeInTheDocument();
    });

    // Check results are displayed
    expect(screen.getByText('2.86e-04')).toBeInTheDocument();
    expect(screen.getByText('64')).toBeInTheDocument(); // batch size
    expect(screen.getByText('64')).toBeInTheDocument(); // lora rank

    // Check explanation
    expect(screen.getByText(/LR = 5e-5 × 10 ×/)).toBeInTheDocument();
  });

  it('handles API errors gracefully', async () => {
    // Mock API error
    mockApi.calculateHyperparameters.mockRejectedValue(
      new Error('Failed to calculate hyperparameters')
    );

    render(<HyperparamCalculator />);

    // Fill and submit form
    fireEvent.change(screen.getByLabelText('Model'), {
      target: { value: 'meta-llama/Llama-3.1-8B' }
    });
    fireEvent.change(screen.getByLabelText('Recipe Type'), {
      target: { value: 'sft' }
    });
    fireEvent.click(screen.getByRole('button', { name: 'Calculate' }));

    // Check error message appears
    await waitFor(() => {
      expect(screen.getByText('Failed to calculate hyperparameters')).toBeInTheDocument();
    });
  });

  it('shows loading state during calculation', async () => {
    // Mock slow API response
    mockApi.calculateHyperparameters.mockImplementation(
      () => new Promise(resolve => setTimeout(resolve, 100))
    );

    render(<HyperparamCalculator />);

    // Fill and submit form
    fireEvent.change(screen.getByLabelText('Model'), {
      target: { value: 'meta-llama/Llama-3.1-8B' }
    });
    fireEvent.change(screen.getByLabelText('Recipe Type'), {
      target: { value: 'sft' }
    });
    fireEvent.click(screen.getByRole('button', { name: 'Calculate' }));

    // Check loading state
    expect(screen.getByText('Calculating...')).toBeInTheDocument();

    // Wait for completion
    await waitFor(() => {
      expect(screen.queryByText('Calculating...')).not.toBeInTheDocument();
    });
  });

  it('validates required fields', () => {
    render(<HyperparamCalculator />);

    // Try to submit without filling required fields
    fireEvent.click(screen.getByRole('button', { name: 'Calculate' }));

    // Should show validation errors
    expect(screen.getByText('Model is required')).toBeInTheDocument();
    expect(screen.getByText('Recipe type is required')).toBeInTheDocument();
  });
});