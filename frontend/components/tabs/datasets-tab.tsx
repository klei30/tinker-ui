'use client';

import { useState } from 'react';
import { Dataset, createDataset } from '@/lib/api';
import { DatasetManager } from '@/components/datasets/dataset-manager';

interface DatasetsTabProps {
  datasets: Dataset[];
  onDatasetCreated: (dataset: Dataset) => void;
  onError: (message: string) => void;
  onSuccess: (message: string) => void;
}

export function DatasetsTab({
  datasets,
  onDatasetCreated,
  onError,
  onSuccess,
}: DatasetsTabProps) {
  const [datasetForm, setDatasetForm] = useState({
    name: '',
    kind: 'huggingface' as Dataset['kind'],
    spec: '{"repo": "allenai/tulu-3-sft-mixture"}',
    description: '',
  });
  const [datasetLoading, setDatasetLoading] = useState(false);

  const handleDatasetSubmit = async (payload: {
    name: string;
    kind: Dataset['kind'];
    spec: Record<string, unknown>;
    description?: string;
  }) => {
    setDatasetLoading(true);
    onError('');
    onSuccess('');
    try {
      const dataset = await createDataset(payload);
      onDatasetCreated(dataset);
      setDatasetForm((prev) => ({ ...prev, name: '', description: '', spec: prev.spec }));
      onSuccess(`Dataset "${dataset.name}" registered successfully.`);
    } catch (error) {
      console.error(error);
      onError((error as Error).message);
    } finally {
      setDatasetLoading(false);
    }
  };

  return (
    <DatasetManager
      datasets={datasets}
      formState={datasetForm}
      submitting={datasetLoading}
      onChange={(updater) =>
        setDatasetForm((prev) => (typeof updater === 'function' ? updater(prev) : { ...prev, ...updater }))
      }
      onSubmit={handleDatasetSubmit}
    />
  );
}