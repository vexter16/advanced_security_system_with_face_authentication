// src/app/api/ask-question/route.ts
import { NextResponse } from 'next/server';

const PYTHON_VQA_API_URL = 'http://localhost:9003/api/ask-question';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // --- THE FIX: Make the validation flexible ---
    const { video_path, temp_filename, question } = body;

    // A question is always required.
    if (!question) {
      return NextResponse.json({ error: 'Missing question' }, { status: 400 });
    }
    
    // EITHER video_path (from dashboard) OR temp_filename (from upload page) must exist.
    if (!video_path && !temp_filename) {
        return NextResponse.json({ error: 'Missing video_path or temp_filename' }, { status: 400 });
    }
    // --- End of fix ---
    
    const pythonResponse = await fetch(PYTHON_VQA_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // Pass the original body along, as it's already in the correct format
      body: JSON.stringify(body), 
      cache: 'no-store',
    });

    if (!pythonResponse.ok) {
      const errorData = await pythonResponse.json().catch(() => ({}));
      return NextResponse.json({ error: 'Error from VQA service', detail: errorData.error || 'Unknown error' }, { status: pythonResponse.status });
    }
    
    if (!pythonResponse.body) {
        return NextResponse.json({ error: 'No response body from backend' }, { status: 500 });
    }

    // Pass the raw text stream directly to the browser
    return new Response(pythonResponse.body, {
      headers: {
        'Content-Type': 'text/plain',
      },
      status: pythonResponse.status,
    });

  } catch (error: any) {
    console.error('[API /ask-question] Error:', error);
    return NextResponse.json({ error: 'Internal Server Error', detail: error.message }, { status: 500 });
  }
}