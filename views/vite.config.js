import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import { zigwui } from '@wui/zigwui/vite';

// Vite's only job here is compiling the Svelte bundle into `dist/`. It never
// serves the console: zignanogpt-web does that, from these compiled bytes.
// There is no dev server: the console refuses a request whose `Host` is not
// its own, and Vite's proxy forwards its own, so a change is `make views` and a
// reload.
//
// The plugins are passed in because the preset is resolved through a symlink
// and cannot find them itself.
export default defineConfig(zigwui({ plugins: [svelte(), tailwindcss()] }));
