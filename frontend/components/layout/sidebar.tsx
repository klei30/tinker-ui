'use client';

import Link from 'next/link';
import { cn } from '@/lib/utils';
import { LayoutDashboard, Play, Database, Layers, MessageSquare } from 'lucide-react';

const NAV_ITEMS = [
  { value: 'overview', label: 'Overview', icon: LayoutDashboard },
  { value: 'runs', label: 'Training', icon: Play },
  { value: 'models', label: 'Models', icon: Layers },
  { value: 'chat', label: 'Chat', icon: MessageSquare },
  { value: 'datasets', label: 'Datasets', icon: Database },
];

interface SidebarProps {
  activeTab?: string;
  onTabChange?: (tab: string) => void;
}

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <div className="flex h-full flex-col bg-muted/20">
      <div className="flex h-16 items-center border-b border-border px-5">
        <Link href="/" className="flex items-center gap-2">
          <img
            src="/favicon.png"
            alt="Tinker Logo"
            className="h-8 w-8 rounded-lg"
          />
          <span className="text-lg font-semibold tracking-tight">Tuner UI</span>
        </Link>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-6">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.value}
              onClick={() => onTabChange?.(item.value)}
              className={cn(
                'flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm font-medium transition-all',
                activeTab === item.value
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="border-t border-border px-4 py-4 text-xs text-muted-foreground">
        <div className="space-y-1">
          <div className="font-medium">Tuner UI</div>
          <div>v0.3</div>
        </div>
      </div>
    </div>
  );
}
