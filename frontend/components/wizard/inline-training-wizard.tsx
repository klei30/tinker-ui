'use client';

import { useState, useEffect } from 'react';
import { Dataset, RecipeType, RunCreatePayload, SupportedModel, getAutoLearningRate, getModelRenderers } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import {
  Sparkles,
  Brain,
  MessageSquare,
  Zap,
  Target,
  Shuffle,
  FlaskConical,
  Calculator,
  Wrench,
  Users,
  Loader2,
  ChevronRight,
  ChevronLeft,
  X,
  Wand2,
  Settings2,
  ChevronDown,
  ChevronUp,
  Rocket,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface InlineTrainingWizardProps {
  isOpen: boolean;
  onClose: () => void;
  datasets: Dataset[];
  supportedModels: SupportedModel[];
  selectedProjectId: number | null;
  onSubmit: (payload: RunCreatePayload) => Promise<void>;
}

// All recipe types with icons and descriptions
const ALL_RECIPES = [
  // ✅ IMPLEMENTED - Popular recipes
  {
    value: 'SFT',
    label: 'Supervised Fine-Tuning',
    shortLabel: 'SFT',
    description: 'Train on instruction-response pairs',
    icon: Zap,
    popular: true,
    supported: true,
    defaultConfig: {
      learning_rate: 5e-5,
      batch_size: 4,
      rank: 64,
      epochs: 3,
    },
  },
  {
    value: 'CHAT_SL',
    label: 'Chat Training',
    shortLabel: 'Chat',
    description: 'Conversational AI training',
    icon: MessageSquare,
    popular: true,
    supported: true,
    defaultConfig: {
      learning_rate: 5e-4,
      batch_size: 64,
      rank: 64,
    },
  },
  {
    value: 'MATH_RL',
    label: 'Math Reasoning RL',
    shortLabel: 'Math RL',
    description: 'Mathematical reasoning with RL',
    icon: Calculator,
    popular: true,
    supported: true,
    defaultConfig: {
      learning_rate: 1e-5,
      rank: 32,
    },
  },

  // ✅ IMPLEMENTED - Advanced recipes
  {
    value: 'DPO',
    label: 'Direct Preference Optimization',
    shortLabel: 'DPO',
    description: 'Learn from preference pairs',
    icon: Target,
    popular: false,
    supported: true,
  },
  {
    value: 'RL',
    label: 'Reinforcement Learning',
    shortLabel: 'RL',
    description: 'General RL training',
    icon: Brain,
    popular: false,
    supported: true,
  },
  {
    value: 'DISTILLATION',
    label: 'Model Distillation',
    shortLabel: 'Distill',
    description: 'Compress model knowledge',
    icon: FlaskConical,
    popular: false,
    supported: true,
  },

  // ❌ COMING SOON - Not yet implemented
  {
    value: 'PPO',
    label: 'Proximal Policy Optimization',
    shortLabel: 'PPO',
    description: 'Advanced RL with PPO',
    icon: Target,
    popular: false,
    supported: false,
    comingSoon: true,
  },
  {
    value: 'GRPO',
    label: 'Group Relative Policy Optimization',
    shortLabel: 'GRPO',
    description: 'Group-based RL optimization',
    icon: Users,
    popular: false,
    supported: false,
    comingSoon: true,
  },
  {
    value: 'PROMPT_DISTILLATION',
    label: 'Prompt Distillation',
    shortLabel: 'P-Distill',
    description: 'Distill prompt strategies',
    icon: Sparkles,
    popular: false,
    supported: false,
    comingSoon: true,
  },
  {
    value: 'TOOL_USE',
    label: 'Tool Use Training',
    shortLabel: 'Tool Use',
    description: 'Train to use external tools',
    icon: Wrench,
    popular: false,
    supported: false,
    comingSoon: true,
  },
  {
    value: 'MULTIPLAYER_RL',
    label: 'Multi-Agent RL',
    shortLabel: 'Multi-RL',
    description: 'Multiple agents training',
    icon: Users,
    popular: false,
    supported: false,
    comingSoon: true,
  },
];

// Recommended models for quick access
const RECOMMENDED_MODELS = [
  'meta-llama/Llama-3.1-8B-Instruct',
  'Qwen/Qwen3-8B-Base',
  'meta-llama/Llama-3.2-3B',
];

export function InlineTrainingWizard({
  isOpen,
  onClose,
  datasets,
  supportedModels,
  selectedProjectId,
  onSubmit,
}: InlineTrainingWizardProps) {
  const [step, setStep] = useState(1);
  const [selectedRecipe, setSelectedRecipe] = useState<RecipeType | ''>('');
  const [selectedModel, setSelectedModel] = useState('meta-llama/Llama-3.1-8B-Instruct');
  const [selectedDataset, setSelectedDataset] = useState('none');
  const [showAllRecipes, setShowAllRecipes] = useState(false);
  const [showAllModels, setShowAllModels] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced options
  const [customHyperparams, setCustomHyperparams] = useState('');
  const [selectedRenderer, setSelectedRenderer] = useState('');
  const [availableRenderers, setAvailableRenderers] = useState<string[]>([]);
  const [wandbProject, setWandbProject] = useState('');
  const [lrScheduler, setLrScheduler] = useState('constant');
  const [warmupSteps, setWarmupSteps] = useState(0);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCalculatingLr, setIsCalculatingLr] = useState(false);

  // Reset when opening
  useEffect(() => {
    if (isOpen) {
      setStep(1);
      setSelectedRecipe('');
      setSelectedModel('meta-llama/Llama-3.1-8B-Instruct');
      setSelectedDataset('none');
      setShowAllRecipes(false);
      setShowAllModels(false);
      setShowAdvanced(false);
      setCustomHyperparams('');
      setWandbProject('');
      setLrScheduler('constant');
      setWarmupSteps(0);
    }
  }, [isOpen]);

  // Load renderers when model changes
  useEffect(() => {
    if (selectedModel && step >= 2) {
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
    }
  }, [selectedModel, step]);

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
    } finally {
      setIsCalculatingLr(false);
    }
  };

  const handleSubmit = async () => {
    if (!selectedProjectId || !selectedRecipe || !selectedModel) return;

    setIsSubmitting(true);
    try {
      const recipe = ALL_RECIPES.find((r) => r.value === selectedRecipe);
      let hyperparameters: Record<string, any> = { ...recipe?.defaultConfig };

      // Merge custom hyperparameters
      if (customHyperparams.trim()) {
        try {
          const customParams = JSON.parse(customHyperparams);
          hyperparameters = { ...hyperparameters, ...customParams };
        } catch (error) {
          alert('Invalid JSON in custom hyperparameters');
          setIsSubmitting(false);
          return;
        }
      }

      // Add renderer
      if (selectedRenderer) {
        hyperparameters.renderer_name = selectedRenderer;
      }

      // Add WandB
      if (wandbProject.trim()) {
        hyperparameters.wandb_project = wandbProject.trim();
      }

      // Add LR scheduler and warmup steps
      if (lrScheduler !== 'constant') {
        hyperparameters.lr_schedule = lrScheduler;
      }
      if (warmupSteps > 0) {
        hyperparameters.lr_warmup_steps = warmupSteps;
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
      onClose();
    } catch (error) {
      console.error('Error creating run:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const popularRecipes = ALL_RECIPES.filter((r) => r.popular);
  const displayedRecipes = showAllRecipes ? ALL_RECIPES : popularRecipes;

  const recommendedModelsList = supportedModels.filter((m) =>
    RECOMMENDED_MODELS.includes(m.model_name)
  );
  const otherModels = supportedModels.filter(
    (m) => !RECOMMENDED_MODELS.includes(m.model_name)
  );
  const displayedModels = showAllModels ? supportedModels : recommendedModelsList;

  const selectedRecipeInfo = ALL_RECIPES.find((r) => r.value === selectedRecipe);

  if (!isOpen) return null;

  return (
    <div className="mb-6 rounded-lg border-2 border-primary/50 bg-card shadow-lg animate-in slide-in-from-top duration-300">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border p-4 bg-primary/5">
        <div className="flex items-center gap-3">
          <Rocket className="h-6 w-6 text-primary" />
          <div>
            <h2 className="text-xl font-bold">New Training Run</h2>
            <p className="text-sm text-muted-foreground">
              {step === 1 && 'Choose your training type'}
              {step === 2 && 'Select model and dataset'}
              {step === 3 && 'Review and launch'}
            </p>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <div className="p-6">
        {/* Step 1: Recipe Selection */}
        {step === 1 && (
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold mb-4">Select Training Type</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {displayedRecipes.map((recipe) => {
                  const RecipeIcon = recipe.icon;
                  const isSelected = selectedRecipe === recipe.value;
                  const isDisabled = recipe.comingSoon || !recipe.supported;

                  return (
                    <button
                      key={recipe.value}
                      onClick={() => !isDisabled && setSelectedRecipe(recipe.value as RecipeType)}
                      disabled={isDisabled}
                      className={cn(
                        'group relative rounded-lg border-2 p-4 text-left transition-all',
                        !isDisabled && 'hover:border-primary/50 hover:shadow-md cursor-pointer',
                        isSelected
                          ? 'border-primary bg-primary/10 shadow-md'
                          : 'border-border bg-card',
                        isDisabled && 'opacity-60 cursor-not-allowed'
                      )}
                    >
                      <div className="flex items-start gap-3">
                        <div
                          className={cn(
                            'rounded-lg p-2',
                            isSelected ? 'bg-primary text-primary-foreground' : 'bg-muted'
                          )}
                        >
                          <RecipeIcon className="h-5 w-5" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <h4 className="font-semibold text-sm">{recipe.shortLabel}</h4>
                            {recipe.comingSoon && (
                              <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4">
                                Soon
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                            {recipe.description}
                          </p>
                        </div>
                      </div>
                      {isSelected && (
                        <div className="absolute top-2 right-2">
                          <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Show All Recipes Toggle */}
              {!showAllRecipes && (
                <Button
                  variant="outline"
                  onClick={() => setShowAllRecipes(true)}
                  className="w-full mt-3"
                >
                  <ChevronDown className="mr-2 h-4 w-4" />
                  Show All {ALL_RECIPES.length} Recipe Types
                </Button>
              )}
              {showAllRecipes && (
                <Button
                  variant="outline"
                  onClick={() => setShowAllRecipes(false)}
                  className="w-full mt-3"
                >
                  <ChevronUp className="mr-2 h-4 w-4" />
                  Show Popular Only
                </Button>
              )}
            </div>
          </div>
        )}

        {/* Step 2: Model & Dataset Selection */}
        {step === 2 && (
          <div className="space-y-6">
            {/* Model Selection */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Choose Model</h3>
              <div className="space-y-2">
                {displayedModels.map((model) => {
                  const isSelected = selectedModel === model.model_name;
                  const isRecommended = RECOMMENDED_MODELS.includes(model.model_name);

                  return (
                    <button
                      key={model.model_name}
                      onClick={() => setSelectedModel(model.model_name)}
                      className={cn(
                        'w-full rounded-lg border-2 p-3 text-left transition-all',
                        'hover:border-primary/50',
                        isSelected
                          ? 'border-primary bg-primary/10'
                          : 'border-border bg-card'
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-sm truncate">
                              {model.model_name}
                            </span>
                            {isRecommended && (
                              <span className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-2 py-0.5 rounded">
                                Recommended
                              </span>
                            )}
                          </div>
                          {model.description && (
                            <p className="text-xs text-muted-foreground mt-1">
                              {model.description}
                            </p>
                          )}
                        </div>
                        {isSelected && (
                          <div className="h-4 w-4 rounded-full bg-primary flex items-center justify-center flex-shrink-0 ml-3">
                            <div className="h-2 w-2 rounded-full bg-white" />
                          </div>
                        )}
                      </div>
                    </button>
                  );
                })}

                {!showAllModels && otherModels.length > 0 && (
                  <Button
                    variant="outline"
                    onClick={() => setShowAllModels(true)}
                    className="w-full"
                  >
                    <ChevronDown className="mr-2 h-4 w-4" />
                    Show All {supportedModels.length} Models
                  </Button>
                )}
                {showAllModels && (
                  <Button
                    variant="outline"
                    onClick={() => setShowAllModels(false)}
                    className="w-full"
                  >
                    <ChevronUp className="mr-2 h-4 w-4" />
                    Show Recommended Only
                  </Button>
                )}
              </div>
            </div>

            {/* Dataset Selection */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Select Dataset (Optional)</h3>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose dataset..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No dataset (use defaults)</SelectItem>
                  {datasets.map((ds) => (
                    <SelectItem key={ds.id} value={ds.id.toString()}>
                      {ds.name} ({ds.kind})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Smart Defaults Preview */}
            {selectedRecipeInfo && (
              <div className="rounded-lg bg-muted/50 p-4 border border-border">
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="h-4 w-4 text-yellow-500" />
                  <h4 className="font-semibold text-sm">Smart Defaults Applied</h4>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {selectedRecipeInfo.defaultConfig && (
                    <>
                      {Object.entries(selectedRecipeInfo.defaultConfig).map(([key, value]) => (
                        <div key={key}>
                          <span className="text-muted-foreground">{key}:</span>
                          <span className="ml-1 font-medium">{String(value)}</span>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Advanced Settings Toggle */}
            <Button
              variant="outline"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="w-full"
            >
              <Settings2 className="mr-2 h-4 w-4" />
              {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
              {showAdvanced ? <ChevronUp className="ml-2 h-4 w-4" /> : <ChevronDown className="ml-2 h-4 w-4" />}
            </Button>

            {/* Advanced Settings */}
            {showAdvanced && (
              <div className="space-y-4 rounded-lg border border-border p-4 bg-muted/20">
                {/* Renderer Selection */}
                {availableRenderers.length > 0 && (
                  <div>
                    <Label>Renderer</Label>
                    <Select value={selectedRenderer} onValueChange={setSelectedRenderer}>
                      <SelectTrigger>
                        <SelectValue />
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

                {/* WandB Project */}
                <div>
                  <Label>Weights & Biases Project (Optional)</Label>
                  <Input
                    placeholder="my-project"
                    value={wandbProject}
                    onChange={(e) => setWandbProject(e.target.value)}
                  />
                </div>

                {/* LR Scheduler */}
                <div>
                  <Label>Learning Rate Schedule</Label>
                  <Select value={lrScheduler} onValueChange={setLrScheduler}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="constant">Constant</SelectItem>
                      <SelectItem value="linear">Linear Decay</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {lrScheduler === 'constant'
                      ? 'Learning rate stays constant throughout training'
                      : 'Learning rate decreases linearly from initial value to 0'}
                  </p>
                </div>

                {/* Warmup Steps */}
                <div>
                  <Label>Warmup Steps (Optional)</Label>
                  <Input
                    type="number"
                    min="0"
                    placeholder="0"
                    value={warmupSteps}
                    onChange={(e) => setWarmupSteps(parseInt(e.target.value) || 0)}
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    Number of steps to gradually increase learning rate from 0 to initial value
                  </p>
                </div>

                {/* Custom Hyperparameters */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <Label>Custom Hyperparameters (JSON)</Label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleAutoLr}
                      disabled={isCalculatingLr}
                    >
                      {isCalculatingLr ? (
                        <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                      ) : (
                        <Wand2 className="mr-2 h-3 w-3" />
                      )}
                      Auto LR
                    </Button>
                  </div>
                  <textarea
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                    rows={6}
                    placeholder={'{\n  "learning_rate": 5e-5,\n  "epochs": 3\n}'}
                    value={customHyperparams}
                    onChange={(e) => setCustomHyperparams(e.target.value)}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-between mt-6 pt-4 border-t border-border">
          <div className="flex gap-2">
            {step > 1 && (
              <Button variant="outline" onClick={() => setStep(step - 1)}>
                <ChevronLeft className="mr-2 h-4 w-4" />
                Back
              </Button>
            )}
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
          </div>

          {step === 1 ? (
            <Button onClick={() => setStep(2)} disabled={!selectedRecipe}>
              Continue
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
          ) : (
            <Button onClick={handleSubmit} disabled={isSubmitting || !selectedRecipe || !selectedModel}>
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Rocket className="mr-2 h-4 w-4" />
                  Start Training
                </>
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
