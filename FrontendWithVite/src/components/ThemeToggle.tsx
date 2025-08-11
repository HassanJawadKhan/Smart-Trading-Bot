import React from 'react';
import { Sun, Moon, Monitor } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';

export const ThemeToggle: React.FC = () => {
  const { theme, actualTheme, toggleTheme } = useTheme();

  const getIcon = () => {
    switch (theme) {
      case 'light':
        return <Sun className="w-5 h-5" />;
      case 'dark':
        return <Moon className="w-5 h-5" />;
      case 'system':
        return <Monitor className="w-5 h-5" />;
      default:
        return <Sun className="w-5 h-5" />;
    }
  };

  const getLabel = () => {
    switch (theme) {
      case 'light':
        return 'Light mode';
      case 'dark':
        return 'Dark mode';
      case 'system':
        return `System (${actualTheme})`;
      default:
        return 'Light mode';
    }
  };

  return (
    <button
      onClick={toggleTheme}
      className="btn-ghost flex items-center gap-2 group"
      title={getLabel()}
      aria-label={getLabel()}
    >
      <span className="transition-transform duration-200 group-hover:scale-110">
        {getIcon()}
      </span>
      <span className="hidden sm:inline text-sm font-medium">
        {theme === 'system' ? 'Auto' : theme === 'light' ? 'Light' : 'Dark'}
      </span>
    </button>
  );
};

export default ThemeToggle;
