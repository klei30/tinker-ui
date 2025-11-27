'use client';

import { useState, useEffect } from 'react';
import { SampleRequest, SampleResponse, sampleModel, ChatRequest, chatWithModel, Run, getRuns, getProjects, Project } from '@/lib/api';
import { ChatConsole } from '@/components/chat/chat-console';

type ChatMessage = { role: 'user' | 'assistant'; content: string };

interface ChatTabProps {
  modelOptions: Array<{ value: string; label: string }>;
  runs: Run[];
  onError: (message: string) => void;
}

export function ChatTab({ modelOptions, runs, onError }: ChatTabProps) {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('Hello! How can I fine-tune a model with Tinker?');
  const [chatModelKey, setChatModelKey] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  // Create options for trained runs - using runs prop which is already being polled
  const trainedRunOptions = runs
    .filter(run => run.status === 'completed')
    .map(run => ({
      value: `run::${run.id}`,
      label: `Run #${run.id} - ${run.recipe_type}`
    }));

  console.log('[Chat] Total runs:', runs.length, 'Completed runs:', trainedRunOptions.length);

  const allModelOptions = [...modelOptions, ...trainedRunOptions];



  const handleChatSubmit = async (prompt: string) => {
    if (!prompt.trim()) return;
    if (!chatModelKey) {
      onError('Select a model for chat sampling.');
      return;
    }
    setChatLoading(true);
    const nextHistory: ChatMessage[] = [...chatHistory, { role: 'user', content: prompt }];
    setChatHistory(nextHistory);
    setChatInput('');
    try {
      const [kind, value] = chatModelKey.split('::');

      if (kind === 'run') {
        // Use trained run for real inference
        const chatRequest: ChatRequest = {
          run_id: Number(value),
          prompt,
          temperature: 0.7,
          max_tokens: 256,
        };
        const response = await chatWithModel(chatRequest);
        const text = response.response;
        setChatHistory((prev) => [...prev, { role: 'assistant', content: `[Real Inference] ${text.trim()}` }]);
      } else if (kind === 'registered') {
        // Use vLLM for real inference with registered models
        try {
          const chatRequest: ChatRequest = {
            prompt,
            model_id: Number(value),
            temperature: 0.7,
            max_tokens: 256,
          };
          const response = await chatWithModel(chatRequest);
          const text = response.response;
          setChatHistory((prev) => [...prev, { role: 'assistant', content: `[Real Inference] ${text.trim()}` }]);
        } catch (error) {
          console.error('vLLM failed, falling back to simulation', error);
          // Fallback to Tinker sampling
          const request: SampleRequest = {
            model_id: Number(value),
            prompt,
            sampling_params: {
              max_tokens: 256,
              temperature: 0.7,
              top_p: 0.9,
              stop: ['User:', 'Assistant:'],
            },
          };
          const response: SampleResponse = await sampleModel(request);
          const text = response.sequences[0]?.text ?? '(no response)';
          setChatHistory((prev) => [...prev, { role: 'assistant', content: `[Simulation] ${text.trim()}` }]);
        }
      } else {
        // Use real inference for base models via chat endpoint
        // The value should be the model_name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        // Extract from "supported::model_name" format
        const modelName = value.replace('supported::', '');
        const chatRequest: ChatRequest = {
          prompt,
          base_model: modelName,
          temperature: 0.7,
          max_tokens: 256,
        };
        const response = await chatWithModel(chatRequest);
        const text = response.response;
        setChatHistory((prev) => [...prev, { role: 'assistant', content: text.trim()}]);
      }
    } catch (error) {
      console.error(error);
      onError((error as Error).message);
    } finally {
      setChatLoading(false);
    }
  };



  return (
    <ChatConsole
      history={chatHistory}
      input={chatInput}
      onInputChange={setChatInput}
      onSubmit={handleChatSubmit}
      loading={chatLoading}
      modelOptions={allModelOptions}
      selectedModelKey={chatModelKey}
      onModelChange={setChatModelKey}
    />
  );
}