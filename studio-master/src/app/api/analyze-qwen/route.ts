// src/app/api/analyze-qwen/route.ts
import { NextResponse } from 'next/server';

const PYTHON_QWEN_API_URL = process.env.PYTHON_ANALYSIS_API_URL?.replace('analyze-video', 'qwen-direct-analysis') || 'http://localhost:9003/api/qwen-direct-analysis';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { video_path } = body;
    if (!video_path) {
      return NextResponse.json({ error: 'Missing video_path' }, { status: 400 });
    }
    
    const pythonResponse = await fetch(PYTHON_QWEN_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_path }),
      // IMPORTANT: Add this for streaming with some environments
      cache: 'no-store',
    });

    if (!pythonResponse.ok) {
      const errorData = await pythonResponse.json().catch(() => ({}));
      return NextResponse.json({ error: 'Error from Qwen analysis service', detail: errorData.error || 'Unknown error' }, { status: pythonResponse.status });
    }
    
    // THE KEY FIX: Instead of awaiting .json(), we pass the stream directly.
    if (!pythonResponse.body) {
        return NextResponse.json({ error: 'No response body from backend' }, { status: 500 });
    }

    // Return a new Response object, using the body from the Python server's response
    return new Response(pythonResponse.body, {
      headers: {
        'Content-Type': 'application/json', // Keep the content type
      },
      status: pythonResponse.status,
    });

  } catch (error: any) {
    console.error('[API /analyze-qwen] Error:', error);
    return NextResponse.json({ error: 'Internal Server Error', detail: error.message }, { status: 500 });
  }
}