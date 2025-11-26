'use client';

import { StatsGrid } from '@/components/overview/stats-grid';
import { OverviewSkeleton } from '@/components/ui/overview-skeleton';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Play, Database, Cpu, MessageSquare } from 'lucide-react';

interface OverviewTabProps {
  onNavigateToRuns: () => void;
  onNavigateToDatasets: () => void;
  onNavigateToModels: () => void;
  onNavigateToChat: () => void;
  stats: Array<{ label: string; value: number }>;
  loading: boolean;
}

export function OverviewTab({
  onNavigateToRuns,
  onNavigateToDatasets,
  onNavigateToModels,
  onNavigateToChat,
  stats,
  loading,
}: OverviewTabProps) {
  if (loading) {
    return <OverviewSkeleton />;
  }

  return (
    <div className="space-y-8">
      <StatsGrid stats={stats} loading={loading} />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        <Card className="cursor-pointer hover:shadow-md transition-all duration-200 hover:shadow-lg border border-border shadow-sm bg-gradient-to-br from-card to-muted/20" onClick={onNavigateToRuns}>
          <CardHeader className="text-center pb-4">
            <div className="mx-auto w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center mb-4">
              <Play className="w-8 h-8 text-primary" />
            </div>
            <CardTitle className="text-xl">Start Training</CardTitle>
            <CardDescription className="text-base">
              Launch a new model training run
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center pt-0">
            <Button size="lg" className="w-full">Create Run</Button>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-all duration-200 hover:shadow-lg border border-border shadow-sm bg-gradient-to-br from-card to-muted/20" onClick={onNavigateToDatasets}>
          <CardHeader className="text-center pb-4">
            <div className="mx-auto w-16 h-16 bg-chart-2/10 rounded-2xl flex items-center justify-center mb-4">
              <Database className="w-8 h-8 text-chart-2" />
            </div>
            <CardTitle className="text-xl">Datasets</CardTitle>
            <CardDescription className="text-base">
              Manage your training data
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center pt-0">
            <Button variant="outline" size="lg" className="w-full">Browse</Button>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-all duration-200 hover:shadow-lg border border-border shadow-sm bg-gradient-to-br from-card to-muted/20" onClick={onNavigateToModels}>
          <CardHeader className="text-center pb-4">
            <div className="mx-auto w-16 h-16 bg-chart-4/10 rounded-2xl flex items-center justify-center mb-4">
              <Cpu className="w-8 h-8 text-chart-4" />
            </div>
            <CardTitle className="text-xl">Models</CardTitle>
            <CardDescription className="text-base">
              Explore available models
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center pt-0">
            <Button variant="outline" size="lg" className="w-full">View</Button>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-all duration-200 hover:shadow-lg border border-border shadow-sm bg-gradient-to-br from-card to-muted/20" onClick={onNavigateToChat}>
          <CardHeader className="text-center pb-4">
            <div className="mx-auto w-16 h-16 bg-chart-5/10 rounded-2xl flex items-center justify-center mb-4">
              <MessageSquare className="w-8 h-8 text-chart-5" />
            </div>
            <CardTitle className="text-xl">Chat</CardTitle>
            <CardDescription className="text-base">
              Test your trained models
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center pt-0">
            <Button variant="outline" size="lg" className="w-full">Start</Button>
          </CardContent>
        </Card>
      </div>

      <Card className="border border-border shadow-sm bg-gradient-to-br from-card to-muted/10">
        <CardHeader className="text-center pb-8">
          <CardTitle className="text-2xl">Getting Started</CardTitle>
          <CardDescription className="text-base mt-2">
            Build your first AI model in four simple steps
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-start gap-6">
            <div className="w-10 h-10 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-semibold text-sm flex-shrink-0">
              1
            </div>
            <div className="pt-1">
              <h4 className="font-semibold text-lg mb-1">Prepare Your Data</h4>
              <p className="text-muted-foreground">
                Upload and organize datasets for training
              </p>
            </div>
          </div>
          <div className="flex items-start gap-6">
            <div className="w-10 h-10 bg-chart-2 text-white rounded-full flex items-center justify-center font-semibold text-sm flex-shrink-0">
              2
            </div>
            <div className="pt-1">
              <h4 className="font-semibold text-lg mb-1">Select a Model</h4>
              <p className="text-muted-foreground">
                Choose from available base models or use your own
              </p>
            </div>
          </div>
          <div className="flex items-start gap-6">
            <div className="w-10 h-10 bg-chart-4 text-white rounded-full flex items-center justify-center font-semibold text-sm flex-shrink-0">
              3
            </div>
            <div className="pt-1">
              <h4 className="font-semibold text-lg mb-1">Configure Training</h4>
              <p className="text-muted-foreground">
                Set parameters and launch your training run
              </p>
            </div>
          </div>
          <div className="flex items-start gap-6">
            <div className="w-10 h-10 bg-chart-5 text-white rounded-full flex items-center justify-center font-semibold text-sm flex-shrink-0">
              4
            </div>
            <div className="pt-1">
              <h4 className="font-semibold text-lg mb-1">Monitor & Test</h4>
              <p className="text-muted-foreground">
                Track progress and interact with your trained model
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}