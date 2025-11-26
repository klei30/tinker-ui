'use client';

import { FormEvent, useState, useMemo } from 'react';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Search, Send } from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatConsoleProps {
  history: ChatMessage[];
  input: string;
  onInputChange: (value: string) => void;
  onSubmit: (prompt: string) => Promise<void> | void;
  loading?: boolean;
  modelOptions: Array<{ value: string; label: string }>;
  selectedModelKey: string;
  onModelChange: (value: string) => void;
}

export function ChatConsole({
  history,
  input,
  onInputChange,
  onSubmit,
  loading,
  modelOptions,
  selectedModelKey,
  onModelChange,
}: ChatConsoleProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!input.trim()) return;
    onSubmit(input.trim());
  };

  // Group and filter models
  const groupedModels = useMemo(() => {
    const trained = modelOptions.filter(m => m.value.startsWith('run::'));
    const registered = modelOptions.filter(m => m.value.startsWith('registered::'));
    const supported = modelOptions.filter(m => m.value.startsWith('supported::'));

    const query = searchQuery.toLowerCase();

    return {
      trained: trained.filter(m => m.label.toLowerCase().includes(query)),
      registered: registered.filter(m => m.label.toLowerCase().includes(query)),
      supported: supported.filter(m => m.label.toLowerCase().includes(query)),
    };
  }, [modelOptions, searchQuery]);

  const selectedModel = modelOptions.find(m => m.value === selectedModelKey);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Chat Playground</CardTitle>
        <p className="text-sm text-muted-foreground">
          Test your trained models and base models
        </p>
      </CardHeader>
      <CardContent className="space-y-4">

        <div className="space-y-2">
          <label className="text-sm font-medium">Model</label>
          <Select value={selectedModelKey} onValueChange={onModelChange}>
            <SelectTrigger>
              <SelectValue placeholder="Select a model..." />
            </SelectTrigger>
            <SelectContent className="max-h-[300px]">
              <div className="sticky top-0 bg-background p-2 border-b">
                <div className="relative">
                  <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-8 h-8 text-sm"
                  />
                </div>
              </div>

              {groupedModels.trained.length > 0 && (
                <SelectGroup>
                  <SelectLabel>Trained Models ({groupedModels.trained.length})</SelectLabel>
                  {groupedModels.trained.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}

              {groupedModels.registered.length > 0 && (
                <SelectGroup>
                  <SelectLabel>Registered Models ({groupedModels.registered.length})</SelectLabel>
                  {groupedModels.registered.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label.replace('Registered · ', '')}
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}

              {groupedModels.supported.length > 0 && (
                <SelectGroup>
                  <SelectLabel>Base Models ({groupedModels.supported.length})</SelectLabel>
                  {groupedModels.supported.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label.replace('Supported · ', '')}
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}

              {groupedModels.trained.length === 0 && groupedModels.registered.length === 0 && groupedModels.supported.length === 0 && (
                <div className="py-4 text-center text-sm text-muted-foreground">
                  No models found
                </div>
              )}
            </SelectContent>
          </Select>
        </div>

        <div className="h-80 overflow-auto rounded border bg-muted/30 p-4">
          {history.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <p className="text-sm text-muted-foreground">
                {selectedModelKey ? "Start a conversation" : "Select a model to begin"}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {history.map((message, index) => (
                <div key={index} className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">
                    {message.role === 'assistant' ? 'Assistant' : 'You'}
                  </div>
                  <div
                    className={
                      message.role === 'assistant'
                        ? 'rounded bg-primary/10 px-3 py-2 text-sm'
                        : 'rounded bg-muted px-3 py-2 text-sm'
                    }
                  >
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  </div>
                </div>
              ))}
              {loading && (
                <p className="text-xs text-muted-foreground">Generating...</p>
              )}
            </div>
          )}
        </div>

        <form className="flex flex-col gap-2" onSubmit={handleSubmit}>
          <Textarea
            className="min-h-20 resize-none"
            value={input}
            onChange={(event) => onInputChange(event.target.value)}
            placeholder={selectedModelKey ? "Type your message..." : "Select a model first"}
            disabled={!selectedModelKey || loading}
          />
          <Button
            type="submit"
            disabled={!selectedModelKey || loading || !input.trim()}
            className="gap-2"
          >
            <Send className="h-4 w-4" />
            {loading ? 'Sending...' : 'Send'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
