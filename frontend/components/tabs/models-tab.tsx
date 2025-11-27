'use client';

import { useState, useMemo } from 'react';
import { ModelCatalogResponse, Dataset, SupportedModel, RegisteredModel, registerModel } from '@/lib/api';
import { ModelCatalog } from '@/components/models/model-catalog';
import { ModelGrid } from '@/components/models/model-grid';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Search, X, LayoutGrid, List } from 'lucide-react';

type ModelRegisterState = { name: string; base_model: string; tinker_path: string; description: string };

interface ModelsTabProps {
  modelCatalog: ModelCatalogResponse | null;
  datasets: Dataset[];
  onError: (message: string) => void;
  onSuccess: (message: string) => void;
  onFineTunePrefill: (model: SupportedModel | RegisteredModel, datasetId?: number) => void;
  onOpenRunsTab: () => void;
}

export function ModelsTab({
  modelCatalog,
  datasets,
  onError,
  onSuccess,
  onFineTunePrefill,
  onOpenRunsTab,
}: ModelsTabProps) {
  const [modelSearch, setModelSearch] = useState('');
  const [registerModelForm, setRegisterModelForm] = useState<ModelRegisterState>({
    name: '',
    base_model: '',
    tinker_path: '',
    description: '',
  });
  const [registeringModel, setRegisteringModel] = useState(false);

  const supportedModels = modelCatalog?.supported_models ?? [];
  const registeredModels = modelCatalog?.registered_models ?? [];

  const handleRegisterModel = async (form: ModelRegisterState) => {
    setRegisteringModel(true);
    onError('');
    onSuccess('');
    try {
      await registerModel({
        name: form.name,
        base_model: form.base_model,
        tinker_path: form.tinker_path || undefined,
        description: form.description || undefined,
        meta: {},
      });
      onSuccess(`Model "${form.name}" registered.`);
      setRegisterModelForm({ name: '', base_model: '', tinker_path: '', description: '' });
    } catch (error) {
      console.error(error);
      onError((error as Error).message);
    } finally {
      setRegisteringModel(false);
    }
  };

  const handleFineTunePrefill = (model: SupportedModel | RegisteredModel, datasetId?: number) => {
    onFineTunePrefill(model, datasetId);
    onOpenRunsTab();
  };

  // View mode state
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  // Filtered models
  const filteredSupported = useMemo(() => {
    if (!modelSearch) return supportedModels;
    return supportedModels.filter((model) =>
      model.model_name.toLowerCase().includes(modelSearch.toLowerCase())
    );
  }, [supportedModels, modelSearch]);

  const filteredRegistered = useMemo(() => {
    if (!modelSearch) return registeredModels;
    return registeredModels.filter((model) => model.name.toLowerCase().includes(modelSearch.toLowerCase()));
  }, [registeredModels, modelSearch]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Model Catalog</h2>
          <p className="text-sm text-muted-foreground">Browse and manage base models & custom checkpoints</p>
        </div>
      </div>

      {/* Search and View Toggle */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search models..."
            value={modelSearch}
            onChange={(e) => setModelSearch(e.target.value)}
            className="pl-9 pr-9"
          />
          {modelSearch && (
            <button
              onClick={() => setModelSearch('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        <div className="flex items-center gap-3">
          <Badge variant="secondary" className="font-mono">
            {filteredSupported.length + filteredRegistered.length} models
          </Badge>

          <div className="flex gap-1 rounded-lg border border-border bg-muted/50 p-1">
            <Button
              variant={viewMode === 'grid' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('grid')}
              className="gap-2"
            >
              <LayoutGrid className="h-4 w-4" />
              <span className="hidden sm:inline">Grid</span>
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('list')}
              className="gap-2"
            >
              <List className="h-4 w-4" />
              <span className="hidden sm:inline">List</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Models Display */}
      {viewMode === 'grid' ? (
        <ModelGrid
          supported={filteredSupported}
          registered={filteredRegistered}
          datasets={datasets}
          onFineTunePrefill={handleFineTunePrefill}
        />
      ) : (
        <ModelCatalog
          supported={filteredSupported}
          registered={filteredRegistered}
          datasets={datasets}
          searchTerm={modelSearch}
          onSearch={setModelSearch}
          registerForm={registerModelForm}
          registering={registeringModel}
          onRegisterChange={(updater) =>
            setRegisterModelForm((prev) => (typeof updater === 'function' ? updater(prev) : { ...prev, ...updater }))
          }
          onRegister={handleRegisterModel}
          onFineTunePrefill={handleFineTunePrefill}
        />
      )}
    </div>
  );
}