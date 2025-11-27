'use client';

import { Dataset, RegisteredModel, SupportedModel } from '@/lib/api';
import { ChangeEvent, FormEvent, useMemo } from 'react';
import { Play } from 'lucide-react';

interface ModelRegisterForm {
  name: string;
  base_model: string;
  tinker_path: string;
  description: string;
}

interface ModelCatalogProps {
  supported: SupportedModel[];
  registered: RegisteredModel[];
  datasets: Dataset[];
  searchTerm: string;
  onSearch: (value: string) => void;
  registerForm: ModelRegisterForm;
  registering?: boolean;
  onRegisterChange: (
    updater: Partial<ModelRegisterForm> | ((prev: ModelRegisterForm) => ModelRegisterForm)
  ) => void;
  onRegister: (payload: ModelRegisterForm) => Promise<void> | void;
  onFineTunePrefill: (model: SupportedModel | RegisteredModel, datasetId?: number) => void;
}

export function ModelCatalog({
  supported,
  registered,
  datasets,
  searchTerm,
  onSearch,
  registerForm,
  registering,
  onRegisterChange,
  onRegister,
  onFineTunePrefill,
}: ModelCatalogProps) {
  const filteredSupported = useMemo(() => {
    if (!searchTerm) return supported;
    return supported.filter((model) => model.model_name.toLowerCase().includes(searchTerm.toLowerCase()));
  }, [supported, searchTerm]);

  const filteredRegistered = useMemo(() => {
    if (!searchTerm) return registered;
    return registered.filter((model) => model.name.toLowerCase().includes(searchTerm.toLowerCase()));
  }, [registered, searchTerm]);

  const handleRegisterSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!registerForm.name.trim() || !registerForm.base_model.trim()) {
      alert('Model name and base model are required.');
      return;
    }
    onRegister(registerForm);
  };

  return (
    <section id="models" className="rounded-lg border border-border bg-card p-6 shadow-sm">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">Model Catalog</h2>
          <p className="mt-1 text-sm text-muted-foreground">Explore base models & checkpoints</p>
        </div>
        <input
          className="h-10 w-full max-w-xs rounded-md border border-border bg-background px-3 text-sm transition focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20"
          placeholder="Search models..."
          value={searchTerm}
          onChange={(event: ChangeEvent<HTMLInputElement>) => onSearch(event.target.value)}
        />
      </div>

      <div className="mt-4 space-y-6">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Supported base models</h3>
          <div className="mt-3 grid gap-3">
            {filteredSupported.map((model) => {
              const nameParts = model.model_name.split('/');
              const hasOrg = nameParts.length > 1;
              const orgName = hasOrg ? nameParts[0] : null;
              const displayName = hasOrg ? nameParts.slice(1).join('/') : model.model_name;

              return (
                <div
                  key={model.model_name}
                  className="group relative rounded-lg border border-border bg-card p-4 transition-all hover:shadow-md hover:border-primary/50"
                >
                  {/* Left accent bar */}
                  <div className="absolute left-0 top-0 bottom-0 w-1 rounded-l-lg bg-gradient-to-b from-blue-500 to-purple-500" />

                  <div className="flex flex-col gap-3 pl-2 md:flex-row md:items-start md:justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start gap-2">
                        <div className="flex-1 min-w-0" title={model.model_name}>
                          {hasOrg && (
                            <div className="text-xs font-medium text-blue-600 dark:text-blue-400 truncate">
                              {orgName}
                            </div>
                          )}
                          <div className="font-semibold text-foreground break-words">
                            {displayName}
                          </div>
                        </div>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted-foreground">
                        <span>{model.parameters ?? 'Unknown'} params</span>
                        <span>•</span>
                        <span>{model.context_length ?? 'Unknown'} ctx</span>
                      </div>
                      {model.description ? (
                        <p className="mt-2 text-sm text-muted-foreground line-clamp-2">{model.description}</p>
                      ) : null}
                    </div>

                    <div className="shrink-0">
                      <button
                        className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition hover:bg-primary/90 whitespace-nowrap group/btn"
                        onClick={() => onFineTunePrefill(model)}
                        title="Start training with this model"
                      >
                        <Play className="h-3.5 w-3.5 transition-transform group-hover/btn:scale-110" />
                        <span>Quick Start</span>
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
            {!filteredSupported.length && (
              <div className="rounded-lg border-2 border-dashed border-border bg-muted/30 p-6 text-center">
                <div className="text-sm text-muted-foreground">No supported models match your search</div>
              </div>
            )}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-foreground">Registered checkpoints</h3>
          <div className="mt-3 grid gap-3">
            {filteredRegistered.map((model) => {
              const nameParts = model.name.split('/');
              const hasOrg = nameParts.length > 1;
              const orgName = hasOrg ? nameParts[0] : null;
              const displayName = hasOrg ? nameParts.slice(1).join('/') : model.name;

              return (
                <div
                  key={model.id}
                  className="group relative rounded-lg border border-border bg-card p-4 transition-all hover:shadow-md hover:border-primary/50"
                >
                  {/* Left accent bar */}
                  <div className="absolute left-0 top-0 bottom-0 w-1 rounded-l-lg bg-gradient-to-b from-green-500 to-emerald-500" />

                  <div className="flex flex-col gap-3 pl-2 md:flex-row md:items-start md:justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start gap-2">
                        <div className="flex-1 min-w-0" title={model.name}>
                          {hasOrg && (
                            <div className="text-xs font-medium text-green-600 dark:text-green-400 truncate">
                              {orgName}
                            </div>
                          )}
                          <div className="font-semibold text-foreground break-words">
                            {displayName}
                          </div>
                        </div>
                      </div>
                      <div className="mt-2 text-xs text-muted-foreground">
                        Base: {model.base_model}
                      </div>
                      {model.tinker_path && (
                        <div className="mt-1 text-[11px] text-muted-foreground font-mono truncate" title={model.tinker_path}>
                          {model.tinker_path}
                        </div>
                      )}
                      {model.description ? (
                        <p className="mt-2 text-sm text-muted-foreground line-clamp-2">{model.description}</p>
                      ) : null}
                    </div>

                    <div className="shrink-0">
                      <button
                        className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition hover:bg-primary/90 whitespace-nowrap group/btn"
                        onClick={() => onFineTunePrefill(model)}
                        title="Start training with this model"
                      >
                        <Play className="h-3.5 w-3.5 transition-transform group-hover/btn:scale-110" />
                        <span>Quick Start</span>
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
            {!filteredRegistered.length && (
              <div className="rounded-lg border-2 border-dashed border-border bg-muted/30 p-6 text-center">
                <div className="text-sm text-muted-foreground">No registered checkpoints yet</div>
              </div>
            )}
          </div>
        </div>

        <div className="relative rounded-lg border border-border bg-card p-6">
          {/* Left accent bar */}
          <div className="absolute left-0 top-0 bottom-0 w-1 rounded-l-lg bg-gradient-to-b from-orange-500 to-amber-500" />

          <div className="pl-2">
            <h3 className="text-base font-semibold text-foreground">Register Checkpoint</h3>
            <p className="mt-1 text-xs text-muted-foreground">Add a custom fine-tuned model to your catalog</p>

            <form className="mt-4 grid gap-3" onSubmit={handleRegisterSubmit}>
              <input
                type="text"
                className="rounded-md border border-border bg-background px-3 py-2 text-sm transition focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20"
                placeholder="Display name"
                value={registerForm.name}
                onChange={(event) => onRegisterChange({ name: event.target.value })}
              />
              <input
                type="text"
                className="rounded-md border border-border bg-background px-3 py-2 text-sm transition focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20"
                placeholder="Base model (e.g. meta-llama/Llama-3.1-8B-Instruct)"
                value={registerForm.base_model}
                onChange={(event) => onRegisterChange({ base_model: event.target.value })}
              />
              <input
                type="text"
                className="rounded-md border border-border bg-background px-3 py-2 text-sm transition focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20"
                placeholder="Checkpoint tinker:// path (optional)"
                value={registerForm.tinker_path}
                onChange={(event) => onRegisterChange({ tinker_path: event.target.value })}
              />
              <textarea
                className="rounded-md border border-border bg-background px-3 py-2 text-sm transition focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20"
                placeholder="Description"
                rows={3}
                value={registerForm.description}
                onChange={(event) => onRegisterChange({ description: event.target.value })}
              />
              <button
                className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
                type="submit"
                disabled={registering}
              >
                {registering ? 'Registering…' : 'Register checkpoint'}
              </button>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
}
