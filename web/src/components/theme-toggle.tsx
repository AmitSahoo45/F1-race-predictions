'use client';

import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  useEffect(() => {
    const saved = localStorage.getItem('apex-theme');
    if (saved === 'light') {
      document.documentElement.dataset.theme = 'light';
      setTheme('light');
    }
  }, []);
  function toggle() {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('apex-theme', next);
    setTheme(next);
  }
  return <button className="theme-toggle" onClick={toggle} type="button" aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
    <span aria-hidden="true">{theme === 'dark' ? '◐' : '◑'}</span><span className="theme-label">{theme === 'dark' ? 'Light' : 'Dark'}</span>
  </button>;
}
