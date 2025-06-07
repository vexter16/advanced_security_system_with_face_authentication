// src/app/api/upload-and-analyze/route.ts
import { NextResponse } from 'next/server';

const PYTHON_UPLOAD_API_URL = process.env.PYTHON_ANALYSIS_API_URL?.replace('analyze-video', 'upload-and-analyze') || 'http://localhost:9003/api/upload-and-analyze';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const videoFile = formData.get('video');

    if (!videoFile) {
      return NextResponse.json({ error: 'No video file found' }, { status: 400 });
    }
    
    const pythonFormData = new FormData();
    pythonFormData.append('video', videoFile as Blob);

    const pythonResponse = await fetch(PYTHON_UPLOAD_API_URL, {
      method: 'POST',
      body: pythonFormData,
      cache: 'no-store',
    });

    if (!pythonResponse.ok) {
      const errorData = await pythonResponse.json().catch(() => ({}));
      return NextResponse.json({ error: 'Error from analysis service', detail: errorData.error || 'Unknown error' }, { status: pythonResponse.status });
    }
    
    // THE SAME FIX: Pass the stream directly through.
    if (!pythonResponse.body) {
      return NextResponse.json({ error: 'No response body from backend' }, { status: 500 });
    }

    return new Response(pythonResponse.body, {
      headers: {
        'Content-Type': 'application/json',
      },
      status: pythonResponse.status,
    });

  } catch (error: any) {
    console.error('[API /upload-and-analyze] Error:', error);
    return NextResponse.json({ error: 'Internal Server Error', detail: error.message }, { status: 500 });
  }
}