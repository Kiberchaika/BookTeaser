// generateAssetManifest.js (ESM)

import { readdirSync, existsSync, mkdirSync, writeFileSync, statSync } from 'fs';
import { join, dirname, relative } from 'path';
import { fileURLToPath } from 'url';

// Define file extensions for images and videos
const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'];
const videoExtensions = ['.mp4', '.webm', '.mov', '.avi', '.ogg'];
const assetExtensions = [...imageExtensions, ...videoExtensions];

const __dirname = dirname(fileURLToPath(import.meta.url));

const publicDir = join(__dirname, 'static');
const outDir = join(__dirname, 'src');
const manifest = [];

// Helper function to test if file is asset type
function isAsset(filename) {
  const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'));
  return assetExtensions.includes(ext);
}

// Recursive walk
function walk(dir) {
  readdirSync(dir).forEach((file) => {
    const filepath = join(dir, file);
    if (statSync(filepath).isDirectory()) {
      walk(filepath); // Recurse
    } else if (isAsset(file) && !file.startsWith('.')) {
      // Create asset URL (remove "public" part, make sure to use / even on Windows)
      const urlPath = '/' + relative(publicDir, filepath).split('\\').join('/');
      manifest.push(urlPath);
    }
  });
}

if (existsSync(publicDir)) {
  walk(publicDir);
}

if (!existsSync(outDir)) {
  mkdirSync(outDir, { recursive: true });
}

writeFileSync(
  join(outDir, 'assets.json'),
  JSON.stringify(manifest, null, 2)
);

console.log('Asset manifest generated:', manifest.length, 'files');
