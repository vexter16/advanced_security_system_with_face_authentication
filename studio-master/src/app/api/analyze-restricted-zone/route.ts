// /full-stack/studio-master/src/app/api/analyze-restricted-zone/route.ts
import { NextResponse } from 'next/server';

const PYTHON_API_URL = 'http://localhost:9003/api/analyze-restricted-zone';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const pythonResponse = await fetch(PYTHON_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
    });

    const result = await pythonResponse.json();
    if (!pythonResponse.ok) {
      return NextResponse.json({ error: result.error || 'Error from analysis service' }, { status: pythonResponse.status });
    }
    return NextResponse.json(result);
  } catch (error: any) {
    return NextResponse.json({ error: 'Internal Server Error', detail: error.message }, { status: 500 });
  }
}