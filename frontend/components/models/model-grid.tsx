'use client';

import { Dataset, RegisteredModel, SupportedModel } from '@/lib/api';
import { cn } from '@/lib/utils';
import {
  Sparkles,
  CheckCircle2,
  ExternalLink,
  Play,
  Database,
  Layers,
  GitBranch,
  Cpu,
  FileText,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';

interface ModelGridProps {
  supported: SupportedModel[];
  registered: RegisteredModel[];
  datasets: Dataset[];
  onFineTunePrefill: (model: SupportedModel | RegisteredModel, datasetId?: number) => void;
}

function ModelCard({
  model,
  type,
  datasets,
  onFineTunePrefill,
}: {
  model: SupportedModel | RegisteredModel;
  type: 'supported' | 'registered';
  datasets: Dataset[];
  onFineTunePrefill: (model: SupportedModel | RegisteredModel, datasetId?: number) => void;
}) {
  const isSupported = type === 'supported';
  const modelName = isSupported ? (model as SupportedModel).model_name : (model as RegisteredModel).name;
  const description = model.description || 'No description available';

  // Split model name into org and model for better display
  const nameParts = modelName.split('/');
  const hasOrg = nameParts.length > 1;
  const orgName = hasOrg ? nameParts[0] : null;
  const displayName = hasOrg ? nameParts.slice(1).join('/') : modelName;

  return (
    <Card className="group relative overflow-hidden transition-all hover:shadow-lg hover:border-primary/50">
      {/* Top accent */}
      <div
        className={cn(
          'absolute left-0 right-0 top-0 h-1',
          isSupported ? 'bg-gradient-to-r from-blue-500 to-purple-500' : 'bg-gradient-to-r from-green-500 to-emerald-500'
        )}
      />

      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <CardTitle className="flex items-start gap-2 text-base leading-tight">
              {isSupported ? (
                <Sparkles className="h-5 w-5 text-blue-500 shrink-0 mt-0.5" />
              ) : (
                <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0 mt-0.5" />
              )}
              <div className="flex-1 min-w-0" title={modelName}>
                {hasOrg && (
                  <div className="text-xs font-normal text-muted-foreground truncate">
                    {orgName}
                  </div>
                )}
                <div className="font-semibold break-words line-clamp-2 leading-snug">
                  {displayName}
                </div>
              </div>
            </CardTitle>
            <CardDescription className="mt-2 line-clamp-2 text-xs">{description}</CardDescription>
          </div>
          <Badge variant={isSupported ? 'default' : 'secondary'} className="shrink-0 self-start">
            {isSupported ? 'Base' : 'Custom'}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3 pb-3">
        {/* Model Info */}
        {isSupported && (model as SupportedModel).parameters && (
          <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <Cpu className="h-3.5 w-3.5" />
              <span>{(model as SupportedModel).parameters} params</span>
            </div>
            {(model as SupportedModel).context_length && (
              <div className="flex items-center gap-1">
                <FileText className="h-3.5 w-3.5" />
                <span>{(model as SupportedModel).context_length} ctx</span>
              </div>
            )}
          </div>
        )}

        {!isSupported && (model as RegisteredModel).base_model && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Layers className="h-3.5 w-3.5" />
            <span>Base: {(model as RegisteredModel).base_model}</span>
          </div>
        )}

        {!isSupported && (model as RegisteredModel).tinker_path && (
          <div className="flex items-center gap-2 text-xs">
            <GitBranch className="h-3.5 w-3.5 text-muted-foreground" />
            <code className="truncate rounded bg-muted px-1.5 py-0.5 text-[10px]">
              {(model as RegisteredModel).tinker_path}
            </code>
          </div>
        )}
      </CardContent>

      <CardFooter className="pt-3 relative z-10">
        {/* Quick Start Button */}
        <button
          className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 group/btn relative z-10"
          onClick={(e) => {
            e.stopPropagation();
            onFineTunePrefill(model);
          }}
          title="Start training with this model"
        >
          <Play className="h-3.5 w-3.5 transition-transform group-hover/btn:scale-110" />
          <span>Quick Start</span>
        </button>
      </CardFooter>

      {/* Hover effect */}
      <div className="absolute inset-x-0 bottom-0 h-0 bg-gradient-to-t from-primary/5 to-transparent transition-all group-hover:h-full pointer-events-none" />
    </Card>
  );
}

export function ModelGrid({ supported, registered, datasets, onFineTunePrefill }: ModelGridProps) {
  return (
    <div className="space-y-8">
      {/* Supported Models */}
      {supported.length > 0 && (
        <section>
          <div className="mb-4 flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-semibold">Base Models</h3>
            <Badge variant="secondary" className="ml-2">
              {supported.length}
            </Badge>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {supported.map((model) => (
              <ModelCard
                key={model.model_name}
                model={model}
                type="supported"
                datasets={datasets}
                onFineTunePrefill={onFineTunePrefill}
              />
            ))}
          </div>
        </section>
      )}

      {/* Registered Models */}
      {registered.length > 0 && (
        <section>
          <div className="mb-4 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-green-500" />
            <h3 className="text-lg font-semibold">Custom Checkpoints</h3>
            <Badge variant="secondary" className="ml-2">
              {registered.length}
            </Badge>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {registered.map((model) => (
              <ModelCard
                key={model.id}
                model={model}
                type="registered"
                datasets={datasets}
                onFineTunePrefill={onFineTunePrefill}
              />
            ))}
          </div>
        </section>
      )}

      {/* Empty state */}
      {supported.length === 0 && registered.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/20 p-12">
          <Layers className="mb-4 h-12 w-12 text-muted-foreground/50" />
          <h3 className="mb-2 text-lg font-semibold">No models found</h3>
          <p className="text-sm text-muted-foreground">Try adjusting your search or filters</p>
        </div>
      )}
    </div>
  );
}
