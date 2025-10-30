import React from 'react';

interface RiskScoreProps {
  score: number;
}

const RiskScore: React.FC<RiskScoreProps> = ({ score }) => {
  const getRiskLevel = (score: number) => {
    if (score < 30) return { level: 'Low', color: 'text-green-600', bg: 'bg-green-100' };
    if (score < 60) return { level: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-100' };
    return { level: 'High', color: 'text-red-600', bg: 'bg-red-100' };
  };

  const risk = getRiskLevel(score);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Risk Score</h2>
      <div className="flex items-center justify-center">
        <div className={`${risk.bg} rounded-full w-32 h-32 flex items-center justify-center`}>
          <div className="text-center">
            <div className={`text-3xl font-bold ${risk.color}`}>{score}</div>
            <div className={`text-sm ${risk.color}`}>{risk.level}</div>
          </div>
        </div>
      </div>
      <div className="mt-4 text-sm text-gray-600 text-center">
        Portfolio risk assessment based on volatility, concentration, and market conditions
      </div>
    </div>
  );
};

export default RiskScore;

