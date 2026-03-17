export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { inspectWorkspaceState, listDocuments } = await import('./_rag.js');
    const [documents, workspace] = await Promise.all([
      listDocuments(),
      inspectWorkspaceState()
    ]);
    res.status(200).json({ documents, workspace });
  } catch (err) {
    let workspace = { apiKeyPresent: false, searchRoots: [], sourceFiles: [] };
    try {
      const { inspectWorkspaceState } = await import('./_rag.js');
      workspace = await inspectWorkspaceState();
    } catch {
      // Keep default workspace diagnostics if inspection fails.
    }
    res.status(200).json({ documents: [], workspace, error: err?.message || 'Document listing failed' });
  }
}
