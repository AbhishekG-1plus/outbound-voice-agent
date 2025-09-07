import asyncio
import logging
from dotenv import load_dotenv
import json
import os
import xml.etree.ElementTree as ET
from time import perf_counter
from typing import Annotated
from datetime import datetime
from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    function_tool,
    Agent,
)
from livekit.agents.voice import AgentSession
from livekit.plugins import deepgram, openai, silero


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

def load_prompt_from_xml(xml_file_path="debtVoiceAgent.xml"):
    """Load system prompt and customer context from XML file"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Extract system prompt
        system_element = root.find('system')
        system_prompt = system_element.text.strip() if system_element is not None else ""
        
        # Extract customer context
        context_element = root.find('context')
        customer_context = ""
        if context_element is not None:
            context_lines = []
            for line in context_element.text.strip().split('\n'):
                line = line.strip()
                if line:
                    context_lines.append(line)
            customer_context = "\n".join(context_lines)
        
        # Combine system prompt with customer context
        full_instructions = system_prompt
        if customer_context:
            full_instructions += f"\n\nCustomer Information:\n{customer_context}"
        
        return full_instructions
    
    except Exception as e:
        logger.warning(f"Failed to load XML prompt: {e}. Using fallback instructions.")
        return (
            "You are a polite and professional payment reminder voice agent. "
            "Your interface with user will be voice. "
            "Be respectful and helpful at all times."
        )

# Load instructions from XML file
_default_instructions = load_prompt_from_xml()

# Add transcript writing callback before connecting
async def write_transcript(phone_number):
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create transcripts directory if it doesn't exist
        os.makedirs("transcripts", exist_ok=True)
        
        # Save to transcripts directory instead of /tmp
        filename = f"transcripts/transcript_{phone_number}_{current_date}.json"
        
        try:
            # Get session from the context if available
            session = getattr(ctx, '_session', None)
            if session and hasattr(session, 'history'):
                with open(filename, 'w') as f:
                    json.dump(session.history.to_dict(), f, indent=2)
                logger.info(f"📄 Transcript for {phone_number} saved to {filename}")
            else:
                logger.warning("No session history available for transcript")
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")

async def entrypoint(ctx: JobContext):
    global _default_instructions, outbound_trunk_id
    logger.info(f"connecting to room {ctx.room.name}")
    user_identity = "phone_user"
    phone_number = ctx.job.metadata
    logger.info(f"dialing {phone_number} to room {ctx.room.name}")
 
    ctx.add_shutdown_callback(write_transcript(phone_number))
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    
    # Use the instructions loaded from XML (already includes customer context)
    instructions = _default_instructions

    # `create_sip_participant` starts dialing the user
    await ctx.api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            room_name=ctx.room.name,
            sip_trunk_id=outbound_trunk_id,
            sip_call_to=phone_number,
            participant_identity=user_identity,
        )
    )

    # a participant is created as soon as we start dialing
    participant = await ctx.wait_for_participant(identity=user_identity)

    await run_voice_pipeline_agent(ctx, participant, instructions)

    # in addition, you can monitor the call status separately
    start_time = perf_counter()
    while perf_counter() - start_time < 30:
        call_status = participant.attributes.get("sip.callStatus")
        if call_status == "active":
            logger.info("user has picked up")
            return
        elif call_status == "automation":
            # if DTMF is used in the `sip_call_to` number, typically used to dial
            # an extension or enter a PIN.
            # during DTMF dialing, the participant will be in the "automation" state
            pass
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
            logger.info("user rejected the call, exiting job")
            break
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
            logger.info("user did not pick up, exiting job")
            break
        await asyncio.sleep(0.1)

    logger.info("session timed out, exiting job")
    ctx.shutdown()


class CallActions(llm.ToolContext):
    """
    Detect user intent and perform actions
    """

    def __init__(
        self, *, api: api.LiveKitAPI, participant: rtc.RemoteParticipant, room: rtc.Room
    ):
        super().__init__(tools=[])
        self.api = api
        self.participant = participant
        self.room = room

    async def hangup(self):
        try:
            await self.api.room.remove_participant(
                api.RoomParticipantIdentity(
                    room=self.room.name,
                    identity=self.participant.identity,
                )
            )
        except Exception as e:
            # it's possible that the user has already hung up, this error can be ignored
            logger.info(f"received error while ending call: {e}")

    @function_tool()
    async def end_call(self):
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {self.participant.identity}")
        await self.hangup()

    @function_tool()
    async def look_up_availability(
        self,
        date: Annotated[str, "The date of the appointment to check availability for"],
    ):
        """Called when the user asks about alternative appointment availability"""
        logger.info(
            f"looking up availability for {self.participant.identity} on {date}"
        )
        await asyncio.sleep(3)
        return json.dumps(
            {
                "available_times": ["1pm", "2pm", "3pm"],
            }
        )

    @function_tool()
    async def confirm_appointment(
        self,
        date: Annotated[str, "date of the appointment"],
        time: Annotated[str, "time of the appointment"],
    ):
        """Called when the user confirms their appointment on a specific date. Use this tool only when they are certain about the date and time."""
        logger.info(
            f"confirming appointment for {self.participant.identity} on {date} at {time}"
        )
        return "reservation confirmed"

    @function_tool()
    async def detected_answering_machine(self):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        logger.info(f"detected answering machine for {self.participant.identity}")
        await self.hangup()


async def run_voice_pipeline_agent(
    ctx: JobContext, participant: rtc.RemoteParticipant, instructions: str
):
    logger.info("starting voice pipeline agent")

    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(
        role="system",
        content=instructions,
    )

    # Create function context with tools
    fnc_ctx = CallActions(api=ctx.api, participant=participant, room=ctx.room)


    # Get the tools from the function context
    tools = []
    for attr_name in dir(fnc_ctx):
        attr = getattr(fnc_ctx, attr_name)
        if hasattr(attr, '__wrapped__') and hasattr(attr.__wrapped__, '_is_function_tool'):
            tools.append(attr)
    
    logger.info(f"tools: {tools}")
    # Create the agent with proper configuration
    agent = Agent(
        instructions=instructions,
        chat_ctx=initial_ctx,
        tools=tools,
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2-phonecall"),
        llm=openai.LLM(),
        tts=openai.TTS(),
    )
    
    session = AgentSession()
    
    # Store session in context for transcript callback
    ctx._session = session

    await session.start(agent, room=ctx.room)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
        raise ValueError(
            "SIP_OUTBOUND_TRUNK_ID is not set"
        )
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # giving this agent a name will allow us to dispatch it via API
            # automatic dispatch is disabled when `agent_name` is set
            agent_name="outbound-caller",
            # prewarm by loading the VAD model, needed only for VoicePipelineAgent
            prewarm_fnc=prewarm,
        )
    )