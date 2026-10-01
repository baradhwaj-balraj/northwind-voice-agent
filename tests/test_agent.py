"""Turn-level tests for the Northwind Fiber support line.

Each test starts a text-only session (no audio, no room), sends what a caller
would say, and checks what the agent did: which tools it called, whether it
handed off to another agent, and whether its reply meant the right thing.

Run with:  uv run pytest
These use the real AI brain through LiveKit Inference, so they need the keys in
.env.local and use a small amount of credit (text only, no voice).
"""

import pytest
from livekit.agents import AgentSession, inference, llm, mock_tools

from agent import BillingAgent, CallRecord, FrontDeskAgent


def _judge_llm() -> llm.LLM:
    # A separate AI that grades whether the agent's reply means what we expect
    return inference.LLM(model="openai/gpt-4.1-mini")


# --- Sam, the front desk -----------------------------------------------------


@pytest.mark.asyncio
async def test_internet_problem_asks_for_zip_not_account() -> None:
    """Regression test: Sam used to demand an account before helping with internet problems."""
    async with _judge_llm() as judge, AgentSession(userdata=CallRecord()) as session:
        await session.start(FrontDeskAgent())

        result = await session.run(user_input="Hi, my internet isn't working.")

        await result.expect.next_event().is_message(role="assistant").judge(
            judge,
            intent=(
                "Asks for the caller's ZIP code (or where they have service) to help "
                "with the internet problem. Does NOT ask for a phone number or account."
            ),
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_outage_zip_reports_outage() -> None:
    """A ZIP code on the outage list should trigger the outage check and an outage explanation."""
    async with _judge_llm() as judge, AgentSession(userdata=CallRecord()) as session:
        await session.start(FrontDeskAgent())

        await session.run(user_input="My internet isn't working.")
        result = await session.run(user_input="My ZIP code is 94107.")

        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(name="check_outage")
        result.expect.next_event().is_function_call_output()
        await result.expect.next_event().is_message(role="assistant").judge(
            judge,
            intent=(
                "Tells the caller there is a known outage in their area and that "
                "service should be restored soon. Does not start router troubleshooting."
            ),
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_already_restarted_creates_ticket() -> None:
    """Regression test: Sam used to say he'd create a ticket without actually creating it."""
    async with _judge_llm() as judge, AgentSession(userdata=CallRecord()) as session:
        await session.start(FrontDeskAgent())

        await session.run(user_input="My internet isn't working.")
        await session.run(user_input="My ZIP code is 10001.")
        result = await session.run(
            user_input="I already restarted my router and it's still not working."
        )

        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(name="create_repair_ticket")
        result.expect.next_event().is_function_call_output()
        await result.expect.next_event().is_message(role="assistant").judge(
            judge,
            intent="Tells the caller a repair ticket was created and gives a ticket number.",
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_incomplete_zip_is_not_used() -> None:
    """Regression test: a four digit ZIP code was once padded to 10000 and saved."""
    async with _judge_llm() as judge, AgentSession(userdata=CallRecord()) as session:
        await session.start(FrontDeskAgent())

        await session.run(user_input="My internet isn't working.")
        result = await session.run(user_input="My ZIP code is one zero zero zero.")

        # Exact check: no made-up ZIP code was saved to the call record
        assert session.userdata.zip_code is None

        await result.expect[-1].is_message(role="assistant").judge(
            judge,
            intent="Asks the caller to repeat or confirm their full five digit ZIP code.",
        )


@pytest.mark.asyncio
async def test_outage_system_down_is_handled_gracefully() -> None:
    """If the outage check fails, Sam should apologize and keep helping, not crash or make things up."""
    async with _judge_llm() as judge, AgentSession(userdata=CallRecord()) as session:
        await session.start(FrontDeskAgent())

        with mock_tools(
            FrontDeskAgent,
            {"check_outage": lambda: RuntimeError("outage system unavailable")},
        ):
            await session.run(user_input="My internet isn't working.")
            result = await session.run(user_input="My ZIP code is 94107.")

        result.expect.contains_function_call(name="check_outage")
        await result.expect[-1].is_message(role="assistant").judge(
            judge,
            intent=(
                "Does not claim to know whether there is an outage. Acknowledges it "
                "could not check, and offers another way to help, such as "
                "troubleshooting or a repair ticket."
            ),
        )


@pytest.mark.asyncio
async def test_billing_question_transfers_to_max() -> None:
    """A billing question should hand the call to Max, the billing specialist."""
    async with AgentSession(userdata=CallRecord()) as session:
        await session.start(FrontDeskAgent())

        result = await session.run(user_input="I have a question about my bill.")

        result.expect.contains_function_call(name="transfer_to_billing")
        result.expect.contains_agent_handoff(new_agent_type=BillingAgent)


# --- Max, billing ------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_does_not_invent_charge_details() -> None:
    """Regression test: the billing agent once made up what a charge was for."""
    async with _judge_llm() as judge, AgentSession(userdata=CallRecord()) as session:
        await session.start(BillingAgent())

        await session.run(user_input="Hi, I have a question about my bill.")
        await session.run(user_input="My phone number is 555 123 4567.")
        await session.run(user_input="Yes, that's correct.")
        result = await session.run(user_input="What exactly is my balance for?")

        await result.expect[-1].is_message(role="assistant").judge(
            judge,
            intent=(
                "Does NOT claim to know what the charge is for (for example, it must "
                "not say it is the monthly service charge). Says it does not have that "
                "detail and offers a follow-up from the billing team."
            ),
        )
