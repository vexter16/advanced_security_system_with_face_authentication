// /full-stack/studio-master/src/app/api/send-alert/route.ts
import { NextResponse } from 'next/server';

const PYTHON_SMS_API_URL = 'http://localhost:9003/api/trigger-sms-alert';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { message } = body;

    if (!message) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    // We don't need to wait for the response from this, just fire it
    fetch(PYTHON_SMS_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    return NextResponse.json({ status: 'Alert triggered' });
  } catch (error: any) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}