'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface LossCurveProps {
  data: Array<{
    step: number;
    train_mean_nll: number;
    learning_rate: number;
    progress: number;
    timestamp: string;
  }>;
}

export function LossCurve({ data }: LossCurveProps) {
  // Transform data for Recharts
  const chartData = data.map((point) => ({
    step: point.step,
    loss: point.train_mean_nll,
    learningRate: point.learning_rate,
    progress: point.progress * 100, // Convert to percentage
  }));

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="step"
            className="text-xs fill-muted-foreground"
            label={{ value: 'Training Step', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            className="text-xs fill-muted-foreground"
            label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            labelStyle={{ color: 'hsl(var(--foreground))' }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={false}
            name="Training Loss"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}