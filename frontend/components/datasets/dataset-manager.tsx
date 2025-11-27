'use client';

import { Dataset } from '@/lib/api';
import { FormEvent } from 'react';

const PRESET_DATASETS: Array<{
  name: string;
  kind: Dataset['kind'];
  spec: Record<string, unknown>;
  description: string;
}> = [
  {
    name: 'allenai/tulu-3-sft-mixture',
    kind: 'huggingface',
    spec: { repo: 'allenai/tulu-3-sft-mixture' },
    description: 'High-quality supervised fine-tuning mixture from Tulu-3',
  },
  {
    name: 'openmath/deepmath-103k',
    kind: 'huggingface',
    spec: { repo: 'openmath/deepmath-103k' },
    description: 'Math reasoning prompts used in distillation recipe',
  },
  {
    name: 'local/chat-traces',
    kind: 'local',
    spec: { path: './data/chat_traces.jsonl' },
    description: 'Upload chat traces stored locally (JSONL)',
  },
  {
    name: 'preference/pairs',
    kind: 'jsonl',
    spec: { path: './data/preference_pairs.jsonl' },
    description: 'Pairwise comparisons for preference learning (JSONL)',
  },
];

interface DatasetManagerProps {
  datasets: Dataset[];
  formState: {
    name: string;
    kind: Dataset['kind'];
    spec: string;
    description: string;
  };
  submitting?: boolean;
  onChange: (
    updater:
      | Partial<DatasetManagerProps['formState']>
      | ((prev: DatasetManagerProps['formState']) => DatasetManagerProps['formState'])
  ) => void;
  onSubmit: (payload: {
    name: string;
    kind: Dataset['kind'];
    spec: Record<string, unknown>;
    description?: string;
  }) => Promise<void> | void;
}

export function DatasetManager({ datasets, formState, submitting, onChange, onSubmit }: DatasetManagerProps) {
  const handlePreset = (preset: (typeof PRESET_DATASETS)[number]) => {
    onChange({
      name: preset.name,
      kind: preset.kind,
      spec: JSON.stringify(preset.spec, null, 2),
      description: preset.description,
    });
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      const spec = JSON.parse(formState.spec);
      onSubmit({
        name: formState.name,
        kind: formState.kind,
        spec,
        description: formState.description || undefined,
      });
    } catch (error) {
      console.error('Invalid dataset spec JSON', error);
      alert('Dataset specification must be valid JSON.');
    }
  };

  return (
    <section id="datasets" className="rounded-2xl border border-border bg-muted/30 p-6 shadow-sm">
      <h2 className="text-lg font-semibold">Datasets</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Register datasets referenced by cookbook recipes. Hugging Face datasets can be referenced by repo slug, local
        datasets via path, and preference datasets via JSONL specifications.
      </p>

      <div className="mt-4 space-y-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Presets</span>
        <div className="flex flex-wrap gap-2">
          {PRESET_DATASETS.map((preset) => (
            <button
              key={preset.name}
              className="rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
              type="button"
              onClick={() => handlePreset(preset)}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <form className="mt-4 flex flex-col gap-3 text-sm" onSubmit={handleSubmit}>
        <label className="flex flex-col gap-1">
          <span>Name</span>
          <input
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={formState.name}
            onChange={(event) => onChange({ name: event.target.value })}
            required
          />
        </label>
        <label className="flex flex-col gap-1">
          <span>Kind</span>
          <select
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={formState.kind}
            onChange={(event) => onChange({ kind: event.target.value as Dataset['kind'] })}
          >
            <option value="huggingface">Hugging Face dataset</option>
            <option value="local">Local path</option>
            <option value="jsonl">JSONL comparisons</option>
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span>Specification (JSON)</span>
          <textarea
            className="h-20 rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
            value={formState.spec}
            onChange={(event) => onChange({ spec: event.target.value })}
            required
          />
        </label>
        <label className="flex flex-col gap-1">
          <span>Description</span>
          <textarea
            className="h-16 rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={formState.description}
            onChange={(event) => onChange({ description: event.target.value })}
          />
        </label>
        <button
          className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
          type="submit"
          disabled={submitting}
        >
          {submitting ? 'Registering…' : 'Register dataset'}
        </button>
      </form>

      <div className="mt-6 space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Registered datasets</div>
        <div className="grid gap-3">
          {datasets.map((dataset) => (
            <div
              key={dataset.id}
              className="rounded-lg border border-border bg-muted/50 p-4 text-xs text-muted-foreground"
            >
              <div className="font-semibold text-foreground">{dataset.name}</div>
              <div className="text-muted-foreground">{dataset.kind}</div>
              <div className="mt-2 overflow-x-auto whitespace-pre text-[10px] leading-4 text-muted-foreground">
                {JSON.stringify(dataset.spec, null, 2)}
              </div>
            </div>
          ))}
          {!datasets.length && (
            <div className="rounded-lg border border-white/10 bg-slate-900/70 p-4 text-xs text-slate-300">
              No datasets registered yet.
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
