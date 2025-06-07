// src/app/api/cleanup-upload/route.ts
import { NextResponse } from 'next/server';

const PYTHON_CLEANUP_API_URL = 'http://localhost:9003/api/cleanup-upload';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // Fire and forget: We don't need to wait for the response
    fetch(PYTHON_CLEANUP_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    return NextResponse.json({ status: 'cleanup_triggered' });

  } catch (error: any) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}