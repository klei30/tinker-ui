'use client';

import { useState, useEffect } from 'react';
import { Dataset, RecipeType, RunCreatePayload, SupportedModel, getAutoLearningRate, getModelRenderers } from '@/lib/api';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Sparkles, Brain, MessageSquare, ChevronDown, ChevronUp, Wand2 } from 'lucide-react';
import { HyperparamCalculator } from '@/components/training/hyperparam-calculator';

interface SimpleRunWizardProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasets: Dataset[];
  supportedModels: SupportedModel[];
  selectedProjectId: number | null;
  onSubmit: (payload: RunCreatePayload) => Promise<void>;
}

// Top 3 most common recipes - simple choices
const QUICK_RECIPES = [
  {
    value: 'SFT',
    label: 'Fine-Tune',
    description: 'Train on your data',
    icon: Sparkles,
    defaultConfig: {
      learning_rate: 5e-5,
      batch_size: 4,
      rank: 64,
      epochs: 3
    }
  },
  {
    value: 'CHAT_SL',
    label: 'Chat Training',
    description: 'Conversational AI',
    icon: MessageSquare,
    defaultConfig: {
      dataset: 'HuggingFaceH4/no_robots',
      learning_rate: 5e-4,
      batch_size: 64,
      rank: 64,
      eval_every: 20,
      save_every: 20
    }
  },
  {
    value: 'MATH_RL',
    label: 'Math RL',
    description: 'Math reasoning',
    icon: Brain,
    defaultConfig: {
      environment: 'arithmetic',
      learning_rate: 1e-5,
      rank: 32,
      group_size: 4,
      groups_per_batch: 100,
      max_tokens: 256
    }
  },
];

// Recommended models for quick access
const RECOMMENDED_MODELS = [
  'meta-llama/Llama-3.1-8B-Instruct',
  'Qwen/Qwen3-8B-Base',
  'meta-llama/Llama-3.2-3B',
];

export function SimpleRunWizard({
  open,
  onOpenChange,
  datasets,
  supportedModels,
  selectedProjectId,
  onSubmit,
}: SimpleRunWizardProps) {
  const [selectedRecipe, setSelectedRecipe] = useState<RecipeType | ''>('');
  const [selectedModel, setSelectedModel] = useState('meta-llama/Llama-3.1-8B-Instruct');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced options
  const [selectedDataset, setSelectedDataset] = useState('none');
  const [wandbProject, setWandbProject] = useState('');
  const [customHyperparams, setCustomHyperparams] = useState('');
  const [selectedRenderer, setSelectedRenderer] = useState('');
  const [availableRenderers, setAvailableRenderers] = useState<string[]>([]);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCalculatingLr, setIsCalculatingLr] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Load available renderers when model changes
  useEffect(() => {
    if (selectedModel) {
      getModelRenderers(selectedModel)
        .then((response) => {
          setAvailableRenderers(response.recommended_renderers);
          setSelectedRenderer(response.default_renderer || '');
        })
        .catch((error) => {
          console.error('Error loading renderers:', error);
          setAvailableRenderers([]);
          setSelectedRenderer('');
        });
    } else {
      setAvailableRenderers([]);
      setSelectedRenderer('');
    }
  }, [selectedModel]);

  const handleAutoLr = async () => {
    if (!selectedModel) return;

    setIsCalculatingLr(true);
    try {
      const result = await getAutoLearningRate(selectedModel, true);
      const currentParams = customHyperparams ? JSON.parse(customHyperparams) : {};
      const updatedParams = { ...currentParams, learning_rate: result.optimal_learning_rate };
      setCustomHyperparams(JSON.stringify(updatedParams, null, 2));
    } catch (error) {
      console.error('Error calculating auto LR:', error);
      alert('Failed to calculate optimal learning rate. Please try again.');
    } finally {
      setIsCalculatingLr(false);
    }
  };

  const handleSubmit = async () => {
    if (!selectedProjectId || !selectedRecipe || !selectedModel) return;

    setIsSubmitting(true);
    try {
      const recipe = QUICK_RECIPES.find(r => r.value === selectedRecipe);
      let hyperparameters: Record<string, any> = { ...recipe?.defaultConfig };

      // Merge custom hyperparameters if provided
      if (customHyperparams.trim()) {
        try {
          const customParams = JSON.parse(customHyperparams);
          hyperparameters = { ...hyperparameters, ...customParams };
        } catch (error) {
          alert('Invalid JSON in custom hyperparameters. Please check your syntax.');
          setIsSubmitting(false);
          return;
        }
      }

      // Add renderer if selected
      if (selectedRenderer) {
        hyperparameters.renderer_name = selectedRenderer;
      }

      // Add WandB project if provided
      if (wandbProject.trim()) {
        hyperparameters.wandb_project = wandbProject.trim();
      }

      const config: any = {
        base_model: selectedModel,
        hyperparameters,
      };

      const payload: RunCreatePayload = {
        project_id: selectedProjectId,
        recipe_type: selectedRecipe,
        config_json: config,
      };

      if (selectedDataset && selectedDataset !== 'none') {
        payload.dataset_id = parseInt(selectedDataset);
      }

      await onSubmit(payload);

      // Reset form
      setSelectedRecipe('');
      setSelectedModel('meta-llama/Llama-3.1-8B-Instruct');
      setSelectedDataset('none');
      setWandbProject('');
      setCustomHyperparams('');
      setSelectedRenderer('');
      setAvailableRenderers([]);
      setShowAdvanced(false);
      onOpenChange(false);
    } catch (error) {
      console.error('Error creating run:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const canSubmit = selectedRecipe && selectedModel && selectedProjectId;

  // Get recommended models that exist in supported list
  const recommendedModelsList = supportedModels.filter(m =>
    RECOMMENDED_MODELS.includes(m.model_name)
  );

  // Get other models
  const otherModelsList = supportedModels.filter(m =>
    !RECOMMENDED_MODELS.includes(m.model_name)
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>New Training Run</DialogTitle>
          <DialogDescription>
            Quick start: Pick a model and training type
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Step 1: Choose Model - Simple Dropdown */}
          <div className="space-y-3">
            <Label className="text-base font-semibold">1. Select Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="w-full h-11">
                <SelectValue placeholder="Choose a model..." />
              </SelectTrigger>
              <SelectContent>
                {recommendedModelsList.length > 0 && (
                  <>
                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                      ⭐ Recommended
                    </div>
                    {recommendedModelsList.map((model) => (
                      <SelectItem key={model.model_name} value={model.model_name}>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{model.model_name.split('/')[1] || model.model_name}</span>
                          <span className="text-xs text-muted-foreground">
                            {model.parameters}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </>
                )}
                {otherModelsList.length > 0 && (
                  <>
                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2">
                      All Models
                    </div>
                    {otherModelsList.map((model) => (
                      <SelectItem key={model.model_name} value={model.model_name}>
                        <div className="flex items-center gap-2">
                          <span>{model.model_name.split('/')[1] || model.model_name}</span>
                          <span className="text-xs text-muted-foreground">
                            {model.parameters}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </>
                )}
              </SelectContent>
            </Select>
          </div>

          {/* Step 2: Choose Recipe - Simple Cards */}
          <div className="space-y-3">
            <Label className="text-base font-semibold">2. Choose Training Type</Label>
            <div className="grid grid-cols-3 gap-3">
              {QUICK_RECIPES.map((recipe) => {
                const Icon = recipe.icon;
                return (
                  <Card
                    key={recipe.value}
                    className={`cursor-pointer transition-all hover:shadow-md p-4 ${
                      selectedRecipe === recipe.value
                        ? 'ring-2 ring-primary bg-primary/5'
                        : 'hover:bg-muted/50'
                    }`}
                    onClick={() => setSelectedRecipe(recipe.value as RecipeType)}
                  >
                    <div className="flex flex-col items-center gap-2 text-center">
                      <Icon className="h-6 w-6 text-primary" />
                      <div className="font-semibold text-sm">{recipe.label}</div>
                      <div className="text-xs text-muted-foreground">{recipe.description}</div>
                    </div>
                  </Card>
                );
              })}
            </div>
          </div>

          {/* Advanced Settings - Collapsed by default */}
          <div className="border-t pt-4">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex w-full items-center justify-between rounded-lg p-3 hover:bg-muted/50 transition-colors"
            >
              <span className="text-sm font-medium flex items-center gap-2">
                {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                Advanced Settings
              </span>
              <span className="text-xs text-muted-foreground">
                {showAdvanced ? 'Hide' : 'Optional'}
              </span>
            </button>

            {showAdvanced && (
              <div className="mt-4 space-y-4 pl-7">
                {/* Dataset Selection */}
                <div className="space-y-2">
                  <Label className="text-sm">Dataset (Optional)</Label>
                  <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                    <SelectTrigger>
                      <SelectValue placeholder="No dataset (use built-in)" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">No dataset (use built-in)</SelectItem>
                      {datasets.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id.toString()}>
                          {dataset.name} <span className="text-muted-foreground">({dataset.kind})</span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* WandB Logging */}
                <div className="space-y-2">
                  <Label className="text-sm">Weights & Biases Project</Label>
                  <Input
                    value={wandbProject}
                    onChange={(e) => setWandbProject(e.target.value)}
                    placeholder="your-project-name (optional)"
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Log training metrics to WandB
                  </p>
                </div>

                {/* Hyperparameter Calculator */}
                <div className="space-y-3">
                  <Label className="text-sm">Hyperparameters</Label>

                  {/* Smart Calculator */}
                  <HyperparamCalculator
                    modelName={selectedModel}
                    recipeType={selectedRecipe || 'SFT'}
                    loraRank={undefined}
                     onApply={(recommendations) => {
                       // Convert recommendations to JSON and apply
                       const params = {
                         learning_rate: recommendations.learning_rate,
                         batch_size: recommendations.batch_size,
                         lora_rank: recommendations.lora_rank,
                         adam_beta1: recommendations.adam_beta1,
                         adam_beta2: recommendations.adam_beta2,
                         adam_eps: recommendations.adam_eps,
                       };
                       setCustomHyperparams(JSON.stringify(params, null, 2));
                     }}
                  />

                  {/* Manual Override */}
                  <div className="space-y-2">
                    <Label className="text-xs text-muted-foreground">
                      Manual Override (JSON)
                    </Label>
                    <textarea
                      value={customHyperparams}
                      onChange={(e) => setCustomHyperparams(e.target.value)}
                      placeholder='{"learning_rate": 1e-5, "batch_size": 4, "rank": 64}'
                      className="w-full h-24 px-3 py-2 text-sm border border-input rounded-md bg-background resize-none font-mono"
                      rows={4}
                    />
                    <p className="text-xs text-muted-foreground">
                      Override any hyperparameters here (optional)
                    </p>
                  </div>
                </div>

                {/* Renderer Selection */}
                {availableRenderers.length > 0 && (
                  <div className="space-y-2">
                    <Label className="text-sm">Chat Template</Label>
                    <Select value={selectedRenderer} onValueChange={setSelectedRenderer}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select renderer..." />
                      </SelectTrigger>
                      <SelectContent>
                        {availableRenderers.map((renderer) => (
                          <SelectItem key={renderer} value={renderer}>
                            {renderer}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Error Display */}
        {submitError && (
          <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
            <p className="text-sm text-destructive font-medium">❌ Error</p>
            <p className="text-sm text-destructive/80">{submitError}</p>
          </div>
        )}

        <div className="flex justify-between items-center pt-4 border-t">
           <div className="text-xs text-muted-foreground">
             {selectedRecipe && selectedModel ? '✓ Ready to start training' : 'Select model and recipe to begin'}
           </div>
           <div className="flex gap-3">
             <Button
               variant="outline"
               onClick={() => onOpenChange(false)}
               disabled={isSubmitting}
             >
               Cancel
             </Button>
             <Button
               onClick={handleSubmit}
               disabled={!canSubmit || isSubmitting}
               className="min-w-[120px]"
             >
               {isSubmitting ? (
                 <>
                   <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                   Creating...
                 </>
               ) : (
                 '🚀 Start Training'
               )}
             </Button>
           </div>
         </div>
      </DialogContent>
    </Dialog>
  );
}
