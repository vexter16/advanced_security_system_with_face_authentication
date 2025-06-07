// src/app/api/analyze-yolo/route.ts
import { NextResponse } from 'next/server';

const PYTHON_YOLO_API_URL = process.env.PYTHON_ANALYSIS_API_URL?.replace('analyze-video', 'yolo-yamnet-analysis') || 'http://localhost:9003/api/yolo-yamnet-analysis';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { video_path } = body;
    if (!video_path) {
      return NextResponse.json({ error: 'Missing video_path' }, { status: 400 });
    }
    
    const pythonResponse = await fetch(PYTHON_YOLO_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_path }),
    });

    if (!pythonResponse.ok) {
      const errorData = await pythonResponse.json().catch(() => ({}));
      return NextResponse.json({ error: 'Error from YOLO/YAMNet analysis service', detail: errorData.error || 'Unknown error' }, { status: pythonResponse.status });
    }
    
    const result = await pythonResponse.json();
    return NextResponse.json(result);

  } catch (error: any) {
    return NextResponse.json({ error: 'Internal Server Error', detail: error.message }, { status: 500 });
  }
}