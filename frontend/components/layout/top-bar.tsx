'use client';

import { ReactNode, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Project, createProject } from '@/lib/api';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { Settings, Cloud } from 'lucide-react';

interface TopBarProps {
  title?: string;
  subtitle?: string;
  projects: Project[];
  selectedProjectId: number | null;
  onSelectProject: (projectId: number) => void;
  onProjectCreated?: (project: Project) => void;
  onError?: (message: string) => void;
  rightSlot?: React.ReactNode;
}

export function TopBar({
  title = 'Tuner UI',
  subtitle,
  projects,
  selectedProjectId,
  onSelectProject,
  onProjectCreated,
  onError,
  rightSlot,
}: TopBarProps) {
  const router = useRouter();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const selectedProjectName = projects.find((p) => p.id === selectedProjectId)?.name;

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return;

    setIsCreating(true);
    try {
      const project = await createProject({
        name: newProjectName.trim(),
        description: newProjectDescription.trim() || undefined,
      });
      onProjectCreated?.(project);
      setNewProjectName('');
      setNewProjectDescription('');
      setIsCreateDialogOpen(false);
    } catch (error) {
      onError?.((error as Error).message);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <header className="flex flex-col border-b border-border bg-card/60 px-6 py-4 lg:px-10">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-xl font-semibold text-foreground lg:text-2xl">{title}</h1>
          {subtitle ? <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p> : null}
        </div>
        <div className="flex flex-col gap-3 text-sm sm:flex-row sm:items-center">
          <div className="flex items-center gap-2 text-sm text-foreground">
            <span className="whitespace-nowrap text-xs uppercase tracking-wide text-muted-foreground">
              Project
            </span>
            <Select
              value={selectedProjectId ? String(selectedProjectId) : undefined}
              onValueChange={(value) => {
                const projectId = Number(value);
                if (!Number.isNaN(projectId)) {
                  onSelectProject(projectId);
                }
              }}
              disabled={!projects.length}
            >
              <SelectTrigger className="w-56">
                <SelectValue placeholder="Select a project..." />
              </SelectTrigger>
              <SelectContent>
                {projects.map((project) => (
                  <SelectItem key={project.id} value={String(project.id)}>
                    {project.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  New Project
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Project</DialogTitle>
                  <DialogDescription>
                    Create a new project to organize your training runs.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="project-name">Project Name</Label>
                    <Input
                      id="project-name"
                      value={newProjectName}
                      onChange={(e) => setNewProjectName(e.target.value)}
                      placeholder="Enter project name"
                    />
                  </div>
                  <div>
                    <Label htmlFor="project-description">Description (Optional)</Label>
                    <Input
                      id="project-description"
                      value={newProjectDescription}
                      onChange={(e) => setNewProjectDescription(e.target.value)}
                      placeholder="Enter project description"
                    />
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button
                      variant="outline"
                      onClick={() => setIsCreateDialogOpen(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handleCreateProject}
                      disabled={isCreating || !newProjectName.trim()}
                    >
                      {isCreating ? 'Creating...' : 'Create Project'}
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => router.push('/settings')}
              className="gap-2"
            >
              <Cloud className="h-4 w-4" />
              <span className="hidden sm:inline">HuggingFace</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push('/deployments')}
              className="gap-2"
            >
              <Settings className="h-4 w-4" />
              <span className="hidden sm:inline">Deployments</span>
            </Button>
            <ThemeToggle />
            {rightSlot}
          </div>
        </div>
      </div>
    </header>
  );
}
