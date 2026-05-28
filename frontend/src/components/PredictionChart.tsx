'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface PredictionChartProps {
  data: Array<{
    class: string;
    probability: number;
  }>;
  maxItems?: number;
}

const COLORS = ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444'];

export default function PredictionChart({ data, maxItems = 10 }: PredictionChartProps) {
  const chartData = data.slice(0, maxItems).map((item, index) => ({
    name: item.class.length > 15 ? item.class.substring(0, 15) + '...' : item.class,
    fullName: item.class,
    probability: (item.probability * 100).toFixed(2),
    value: item.probability,
  }));

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#475569" horizontal={false} />
          <XAxis 
            type="number" 
            domain={[0, 100]}
            tick={{ fill: '#94a3b8', fontSize: 12 }}
            tickFormatter={(value) => `${value}%`}
          />
          <YAxis 
            type="category" 
            dataKey="name" 
            tick={{ fill: '#94a3b8', fontSize: 12 }}
            width={90}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '8px',
              color: '#f1f5f9',
            }}
          />
          <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
