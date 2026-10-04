import { FlatCompat } from '@eslint/eslintrc';

const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const config = [
  { ignores: ['.next/**', 'out/**', 'node_modules/**', 'src/generated/**', 'public/**', 'next-env.d.ts'] },
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
  // Lighthouse loads its configuration as CommonJS.
  { files: ['lighthouserc.cjs'], rules: { '@typescript-eslint/no-require-imports': 'off' } },
];

export default config;
