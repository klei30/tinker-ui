'use client';

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ProgressChartProps {
  data: Array<{
    step: number;
    train_mean_nll: number;
    learning_rate: number;
    progress: number;
    timestamp: string;
  }>;
}

export function ProgressChart({ data }: ProgressChartProps) {
  // Transform data for Recharts
  const chartData = data.map((point) => ({
    step: point.step,
    progress: point.progress * 100, // Convert to percentage
    learningRate: point.learning_rate,
  }));

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="step"
            className="text-xs fill-muted-foreground"
            label={{ value: 'Training Step', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            domain={[0, 100]}
            className="text-xs fill-muted-foreground"
            label={{ value: 'Progress (%)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            labelStyle={{ color: 'hsl(var(--foreground))' }}
            formatter={(value: number, name: string) => [
              name === 'progress' ? `${value.toFixed(1)}%` : value.toFixed(6),
              name === 'progress' ? 'Progress' : 'Learning Rate'
            ]}
          />
          <Area
            type="monotone"
            dataKey="progress"
            stroke="hsl(var(--primary))"
            fill="hsl(var(--primary))"
            fillOpacity={0.3}
            name="progress"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}