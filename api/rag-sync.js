import { syncWorkspaceDocuments } from './_rag.js';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const result = await syncWorkspaceDocuments();
    res.status(200).json({ ok: true, ...result });
  } catch (err) {
    res.status(400).json({ error: err.message || 'Workspace sync failed' });
  }
}
