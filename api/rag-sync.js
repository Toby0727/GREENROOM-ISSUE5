export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { inspectWorkspaceState, syncWorkspaceDocuments } = await import('./_rag.js');
    const result = await syncWorkspaceDocuments();
    let workspace = { apiKeyPresent: false, searchRoots: [], sourceFiles: [] };
    try {
      workspace = await inspectWorkspaceState();
    } catch {
      // Keep default workspace diagnostics if inspection fails.
    }
    res.status(200).json({ ok: true, ...result, workspace });
  } catch (err) {
    let workspace = { apiKeyPresent: false, searchRoots: [], sourceFiles: [] };
    try {
      const { inspectWorkspaceState } = await import('./_rag.js');
      workspace = await inspectWorkspaceState();
    } catch {
      // Keep default workspace diagnostics if inspection fails.
    }
    res.status(400).json({ error: err?.message || 'Workspace sync failed', workspace });
  }
}
