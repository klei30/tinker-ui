'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface StatCardProps {
  label: string;
  value: number;
  loading?: boolean;
}

function StatCard({ label, value, loading }: StatCardProps) {
  return (
    <Card className="border border-border shadow-sm bg-gradient-to-br from-card to-muted/20">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="text-3xl font-bold text-foreground">
          {loading ? <span className="animate-pulse text-muted-foreground">—</span> : value}
        </div>
      </CardContent>
    </Card>
  );
}

interface StatsGridProps {
  stats: Array<{ label: string; value: number }>;
  loading?: boolean;
}

export function StatsGrid({ stats, loading }: StatsGridProps) {
  return (
    <section id="overview" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => (
        <StatCard key={stat.label} label={stat.label} value={stat.value} loading={loading} />
      ))}
    </section>
  );
}
