import React from 'react';
import { Activity } from 'lucide-react';

interface Props {
  greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
  } | null;
}

const GreeksDisplay: React.FC<Props> = ({ greeks }) => {
  if (!greeks) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          Portfolio Greeks
        </h2>
        <p className="text-gray-500">No Greeks data available</p>
      </div>
    );
  }

  const greeksList = [
    {
      name: 'Delta',
      value: greeks.delta,
      description: 'Price sensitivity',
      color: 'blue',
    },
    {
      name: 'Gamma',
      value: greeks.gamma,
      description: 'Delta change rate',
      color: 'green',
    },
    {
      name: 'Theta',
      value: greeks.theta,
      description: 'Time decay',
      color: 'red',
    },
    {
      name: 'Vega',
      value: greeks.vega,
      description: 'IV sensitivity',
      color: 'purple',
    },
    {
      name: 'Rho',
      value: greeks.rho,
      description: 'Rate sensitivity',
      color: 'yellow',
    },
  ];

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center">
        <Activity className="w-5 h-5 mr-2" />
        Portfolio Greeks
      </h2>

      <div className="space-y-4">
        {greeksList.map((greek) => (
          <div key={greek.name} className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center">
                <span className="font-medium text-gray-900">{greek.name}</span>
                <span className="text-sm text-gray-500 ml-2">
                  {greek.description}
                </span>
              </div>
              <div className="mt-1">
                <div className="flex items-center">
                  <div className="flex-1 bg-gray-200 rounded-full h-2 mr-4">
                    <div
                      className={`bg-${greek.color}-600 h-2 rounded-full`}
                      style={{
                        width: `${Math.min(Math.abs(greek.value) / 100 * 100, 100)}%`,
                      }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-gray-900 w-20 text-right">
                    {greek.value.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GreeksDisplay;

