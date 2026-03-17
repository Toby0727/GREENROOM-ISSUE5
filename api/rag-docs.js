import { inspectWorkspaceState, listDocuments } from './_rag.js';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const [documents, workspace] = await Promise.all([
      listDocuments(),
      inspectWorkspaceState()
    ]);
    res.status(200).json({ documents, workspace });
  } catch (err) {
    const workspace = await inspectWorkspaceState();
    res.status(200).json({ documents: [], workspace, error: err.message || 'Document listing failed' });
  }
}
