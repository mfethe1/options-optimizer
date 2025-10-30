import React from 'react';
import { Plus, RefreshCw, FileText, Settings } from 'lucide-react';

const QuickActions: React.FC = () => {
  const actions = [
    { icon: Plus, label: 'Add Position', href: '/positions' },
    { icon: RefreshCw, label: 'Refresh Data', onClick: () => window.location.reload() },
    { icon: FileText, label: 'Generate Report', href: '/reports' },
    { icon: Settings, label: 'Settings', href: '/settings' },
  ];

  return (
    <div className="flex gap-2">
      {actions.map((action, index) => {
        const Icon = action.icon;
        const content = (
          <>
            <Icon className="w-4 h-4" />
            <span className="hidden sm:inline">{action.label}</span>
          </>
        );

        if (action.href) {
          return (
            <a
              key={index}
              href={action.href}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              {content}
            </a>
          );
        }

        return (
          <button
            key={index}
            onClick={action.onClick}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            {content}
          </button>
        );
      })}
    </div>
  );
};

export default QuickActions;

