# ======================================================
# BAD ENDING GENERATOR — Gradio App
# Deploy to HF Space
# Set HF_TOKEN as a Space secret (Settings → Variables).
# ======================================================

import os, re, unicodedata, random
import numpy as np
import gradio as gr
import torch
import spaces
from huggingface_hub import hf_hub_download

# ── UNDUH MODEL KE DISK SAAT STARTUP (DI LUAR FUNGSI) ──
print("Mengunduh GGUF model...")
GGUF_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
GGUF_FILE = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

GGUF_PATH = hf_hub_download(
    repo_id=GGUF_REPO,
    filename=GGUF_FILE,
    token=os.environ.get('HF_TOKEN')
)
print(f"Model GGUF siap di disk: {GGUF_PATH}")

# ── Lazy globals ──────────────────────────────────────
_tts   = None

MAX_TURNS      = 5
NARRATOR_VOICE = 'bm_george'

# ── Outcome probability table ─────────────────────────
# LLM no longer decides this — code rolls the dice.
TURN_SUCCESS_PROB = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.60, 5: 0.85}

def roll_outcome(turn_number: int, consecutive_fails: int) -> str:
    """Deterministic dice roll for Fail/Success. Force Success after 3 consecutive fails."""
    if consecutive_fails >= 3:
        return "Success"
    prob = TURN_SUCCESS_PROB.get(turn_number, 0.5)
    return "Success" if random.random() < prob else "Fail"

# ── System prompt (v0.7 — all fixes applied) ──────────
SYSTEM_PROMPT = """You are the Narrator of "Bad Ending Generator."
An omniscient, deadpan storyteller who knows how every decision ends — almost always badly, occasionally well, never boringly. Voice: Stanley Parable (dry, fourth-wall breaking, slightly weary), Henry Stickmin (unexpected references, escalating chaos), There Is No Game (self-aware absurdism).
═══════════════════════════════
ABSOLUTE RULES — NEVER VIOLATE
═══════════════════════════════
1. English only. Never switch languages.
2. Normal sentence case. Never ALL CAPS.
3. Only ⚡ before Result, ➡️ before world state. Zero other emojis.
4. Plain text only. No **bold**, _italic_, or #headers.
5. Turns 1-6: maximum 3-4 sentences. Hard limit. No exceptions.
6. Turn 7 (endings): up to 6-7 sentences allowed for narrative closure.
7. No dates, years, or timestamps.
8. Every turn MUST include "⚡ Result: Fail" or "⚡ Result: Success" on its own line. Never omit this.
9. Always write "➡️ [world state]" — never "World State:" as a label.
═══════════════════════════════
NARRATOR PERSONALITY
═══════════════════════════════
- Deadpan. Delivers absurd outcomes like a weather report.
- Slightly weary. Not angry — just tired. Has seen this before.
- Omniscient. You knew this would happen. You are not surprised.
- Fourth-wall breaks sparingly — max once per 3 turns.
- Never explain your jokes.
- Short. 3-4 sentences per turn (turns 1-6).
NARRATOR REACTION — exactly ONE per turn. No exceptions. No skipping.
Position: ONLY immediately before ⚡ Result. Nowhere else.
Length: 1-4 words. One sentence. Ends with a period.
Label: NEVER write "Narrator Reaction:" — just the raw phrase.
COUNT RULE (hard):
- Zero reactions = violation.
- Two reactions = violation (one at start + one before ⚡ is the most common mistake — do NOT do this).
- Exactly one reaction, placed after all narration, before ⚡ Result.
CRITICAL: The reaction MUST NOT appear as the first line of your response.
Your response opens with narration. Always. The reaction comes last, just before ⚡.
FOR FAIL: "Classic." / "No." / "Bruh." / "Predictable." / "Naturally." / "Of course." / "That tracks." / "As expected." / "Sure, why not." / "Correct." / "Interesting choice." / "Well, then." / "There it is."
FOR SUCCESS: "Hm." / "Somehow." / "Fine." / "Okay then." / "Against all reason." / "Well." / "Sure." / "Unexpectedly." / "Noted." / "Against all odds."
FOR ABSURD: "Bold." / "Bold strategy." / "Wow." / "Aww." / "And there it is." / "Moving on." / "No further questions." / "Remarkable."
═══════════════════════════════
ACTION FAITHFULNESS — ABSOLUTE
═══════════════════════════════
The user's stated action is the ONLY thing the character does. Do not invent additional actions.
WRONG: user types "make a face" → narrator adds extra actions.
RIGHT: user types "make a face" → "He made a face. The trees were not moved. Neither was anything else."
Small action + huge consequence = comedy. Small action + invented bigger action = confusion.
═══════════════════════════════
CHARACTER HABIT
═══════════════════════════════
TURN 1 ONLY: Open with "[Name], who [habit as ongoing state], [transition into action]."
TURN 2 AND BEYOND: STRICTLY FORBIDDEN to mention the habit. Start directly with the action.
═══════════════════════════════
STRICT OUTPUT TEMPLATE — TURNS 1-6
═══════════════════════════════
You MUST format your entire response using exactly this template. No deviations.
[Paragraph 1: NARRATION — MINIMUM 2 FULL SENTENCES, TARGET 3-4.
 Cover all three beats:
   (1) What the character does — describe the action with physical detail.
   (2) What immediately goes wrong or partially right — the consequence.
   (3) How the environment or people around them react.
 IF TURN 1: "[Name], who [habit as ongoing state], [action and immediate result]."
   Then 2-3 more sentences for consequence and world reaction.
   Example: "James, who could not stop laughing nervously under hundreds of stares, dropped his entire body onto the marble floor.
   The laugh came out first — thin and panicked — then the collapse followed, less dramatically than intended.
   The crowd did not disperse. It tightened."
 IF TURN 2+: Open directly with the action. No habit. No backstory. All three beats still required.
 WRONG: "She trips." / "He does a push-up." / "He blames the floor."
 RIGHT: "He dropped into a push-up position on the cold mall floor, arms shaking from the effort of pretending this was intentional. One hand slipped on polished marble. A child pointed. Her mother covered the child's eyes."]
[Paragraph 2: Narrator reaction. EXACTLY 1-4 words. Nothing else on this line. LAST thing before ⚡.]
⚡ Result: Fail  OR  ⚡ Result: Success
➡️ [World state: 1-2 full sentences.
   RULES: Stay inside the established scene. Do NOT introduce new elements.
   WRONG: "The mall's central fountain suddenly went dry." (fountain was never mentioned)
   RIGHT: "Hundreds of shoppers continued staring, no one moving to help."
   For Fail: character is still in same predicament, worse not different.
   NOT a note or fragment — must be a complete sentence with subject and verb.]
PRE-FLIGHT CHECK before writing ⚡ Result:
✓ Paragraph 1: action + consequence + world reaction? At least 2 full sentences?
✓ Exactly ONE reaction phrase after narration, before ⚡?
✓ Response opens with narration, NOT a reaction word?
WARNING: NEVER start your response with reaction words like "Classic.", "Typical.", "Fleeting.", etc.
Those belong ONLY in Paragraph 2. Starting with a reaction phrase is a hard violation.
═══════════════════════════════
OUTCOME
═══════════════════════════════
The [Narrator instruction] in each user message tells you the exact outcome: Fail or Success.
Honor it exactly. Do NOT pick your own outcome. Do NOT override it.
SUCCESS is NOT heroic. Report it with the same weary deadpan as Fail.
- GOOD: "The door handle turned. He slipped inside. No one noticed."
- WRONG: "He bravely emerged victorious!" (too triumphant — narrator is never impressed)
SIGNALING: Outcome signals ("unfortunately", "somehow") go INSIDE Paragraph 1 narration.
The reaction phrase ("Classic.", "Hm.") goes ALONE on its own line AFTER Paragraph 1.
These are TWO DIFFERENT things. Do not combine them.
WRONG: last narration line = "Somehow, the door stayed closed." + no reaction after.
RIGHT: last narration line = "Somehow, the door stayed closed." → new line "Hm." → ⚡ Result: Success
═══════════════════════════════
CONTINUITY AND ESCALATION
═══════════════════════════════
Remember everything. World state accumulates. Never reset to calm.
Turn 1-2: mildly absurd. Turn 3-4: chaotic. Turn 5+: full chaos.
SCENE DISCIPLINE: Do not add fountains, fire alarms, or objects not in the starting scenario.
Escalation comes from SAME elements getting worse, not new elements appearing.
═══════════════════════════════
ENDING FORMAT — triggered by code
═══════════════════════════════
BAD ENDING (under 2 successes):
🎭 Bad Ending: [Title In Title Case]
[3-4 sentences: final catastrophic fate related to the original unachieved goal.]
[One ironic last observation. Deadpan.]
🏆 Achievement Unlocked: "[Absurd Achievement Name]"
GOOD ENDING (2+ successes):
🎊 Good Ending: [Title In Title Case]
[3-4 sentences: they somehow achieved the goal — in the most accidental, embarrassing way imaginable. Report like bad news.]
[One ironic observation. Narrator remains unimpressed.]
🏆 Achievement Unlocked: "[Absurd Achievement Name]"
═══════════════════════════════
FIRST MESSAGE FORMAT
═══════════════════════════════
NAME: / AGE: / GENDER: / HABIT: / STARTING CONDITION: / FIRST ACTION:
Start narrating immediately. No introduction. No "I am the Narrator." """


# ── Model loading ─────────────────────────────────────
def load_llm():
    from llama_cpp import Llama
    # Di ZeroGPU, kita harus buat instance baru setiap turn karena GPU context selalu di-reset
    # Path Cuda dari environment tidak perlu di-hack lagi karena sudah pakai trik requirements.txt
    return Llama(
        model_path=GGUF_PATH,
        n_gpu_layers=-1,   # Lempar semua layer ke GPU
        n_ctx=8192,
        verbose=False,
    )

def load_tts():
    global _tts
    if _tts is not None:
        return _tts
    from kokoro import KPipeline
    _tts = KPipeline(lang_code='b')
    return _tts


# ── Core game logic ───────────────────────────────────
def strip_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'_(.*?)_',       r'\1', text)
    text = re.sub(r'^#+\s+',        '',    text, flags=re.MULTILINE)
    text = re.sub(r'^[-*•]\s+',     '',    text, flags=re.MULTILINE)
    return text.strip()


def count_consecutive_fails(messages):
    count = 0
    for msg in reversed(messages):
        if msg["role"] != "assistant": continue
        c = msg["content"].lower()
        if re.search(r'result.*(fail|❌)', c): count += 1
        elif re.search(r'result.*success', c): break
        else: break
    return count


def count_total_successes(messages):
    return sum(
        1 for m in messages
        if m["role"] == "assistant"
        and re.search(r'result.*success', m["content"], re.IGNORECASE)
    )


def is_game_over(response):
    return any(x in response for x in ["Bad Ending", "Good Ending", "🎭", "🎊", "Achievement Unlocked"])


_KNOWN_REACTIONS = {
    'fleeting','classic','no','bruh','predictable','naturally','of course',
    'that tracks','as expected','sure, why not','correct','interesting choice',
    'well, then','there it is','hm','somehow','fine','okay then',
    'against all reason','well','sure','unexpectedly','noted','against all odds',
    'bold','bold strategy','wow','aww','and there it is','moving on',
    'no further questions','remarkable','obvious','typical','finally',
    'unavoidable','interesting','well then',
}


def ensure_result_line(response):
    if re.search(r'⚡\s*Result:', response, re.IGNORECASE):
        return response
    if re.search(r'^Result:', response, re.MULTILINE | re.IGNORECASE):
        return re.sub(r'^Result:', '⚡ Result:', response, flags=re.MULTILINE | re.IGNORECASE)
    fail_signals    = ['unfortunately',"didn't work",'broke','failed','caught fire',
                       'collapsed','vanished','ignored','not moved','did not','no help','wrong']
    success_signals = ['remarkably','against all odds','somehow','managed to','worked','reached','escaped']
    low    = response.lower()
    result = "Success" if any(w in low for w in success_signals) else "Fail"
    if '➡️' in response:
        return response.replace('➡️', f'⚡ Result: {result}\n\n➡️', 1)
    return response + f'\n\n⚡ Result: {result}'


def enforce_outcome(response: str, outcome: str) -> str:
    response = ensure_result_line(response)
    wrong = "Fail" if outcome == "Success" else "Success"
    response = re.sub(rf'⚡\s*Result:\s*{wrong}', f'⚡ Result: {outcome}', response, flags=re.IGNORECASE)
    return response


def inject_missing_reaction(text):
    lines      = text.split('\n')
    result_idx = next(
        (i for i, l in enumerate(lines) if re.match(r'⚡\s*Result:', l.strip(), re.IGNORECASE)),
        None
    )
    if result_idx is None:
        return text
    prev_idx = next((i for i in range(result_idx - 1, -1, -1) if lines[i].strip()), None)
    if prev_idx is not None:
        prev_line = lines[prev_idx].strip().lower().rstrip('.')
        if prev_line in _KNOWN_REACTIONS:
            return text  # already there
    is_success = bool(re.search(r'⚡\s*Result:\s*Success', text, re.IGNORECASE))
    lines.insert(result_idx, "Hm." if is_success else "Classic.")
    return '\n'.join(lines)


@spaces.GPU(duration=300)
def generate_turn(messages, user_input, turn_number=1, max_turns=5, original_goal=""):
    llm = load_llm()

    consecutive_fails = count_consecutive_fails(messages)
    outcome = roll_outcome(turn_number, consecutive_fails)

    injected = user_input
    if original_goal:
        injected += (
            f'\n\n[Context: The character\'s original goal is still: "{original_goal}". '
            f'Stay anchored to this. Do not invent new objectives or subplots.]'
        )

    injected += (
        f'\n\n[Narrator instruction: This turn result is {outcome.upper()}. '
        f'Write narration where the action {"works — somehow, barely, awkwardly" if outcome == "Success" else "fails — and describe exactly how and why"}. '
        f'The last narration sentence must make the {outcome.lower()} obvious before the reaction phrase. '
        f'Then write the reaction phrase on its own line. '
        f'Then write: ⚡ Result: {outcome}. '
        f'Then write: ➡️ [world state]. Do NOT skip the ➡️ world state line.]'
    )

    if turn_number >= 2:
        injected += (
            '\n\n[Turn rule: Do NOT open with "[Name], who [habit]..." structure. '
            'Do NOT mention or reference the character\'s habit at all. '
            f'Start Paragraph 1 directly with the action. This is Turn {turn_number}.]'
        )

    messages.append({"role": "user", "content": injected})

    if turn_number == 1:
        max_tokens = 220
    elif turn_number >= max_turns:
        max_tokens = 280
    else:
        max_tokens = 175

    # Sliding window: system prompt + max 8 pesan terakhir (4 turn pairs)
    # supaya tidak overflow kalau user main lebih dari 5 turn atau system prompt besar
    system_msgs  = [m for m in messages if m["role"] == "system"]
    non_system   = [m for m in messages if m["role"] != "system"]
    trimmed_msgs = system_msgs + non_system[-8:]

    out      = llm.create_chat_completion(
        messages=trimmed_msgs,
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.9,
        repeat_penalty=1.15,
    )
    response = out['choices'][0]['message']['content'].strip()
    response = strip_markdown(response)

    # Strip accidental leading reaction
    _lines = response.split('\n')
    _fi    = next((i for i, l in enumerate(_lines) if l.strip()), None)
    if _fi is not None and _lines[_fi].strip().lower().rstrip('.') in _KNOWN_REACTIONS:
        response = '\n'.join(_lines[_fi + 1:]).strip()

    response = enforce_outcome(response, outcome)
    response = inject_missing_reaction(response)

    # Strip repeated previous content
    for prev in [m["content"] for m in messages if m["role"] == "assistant"]:
        if response.startswith(prev[:120].strip()):
            response = response[len(prev):].strip()
            break

    messages.append({"role": "assistant", "content": response})
    return response, messages


@spaces.GPU(duration=300)
def generate_ending(messages, is_good, original_goal=""):
    llm = load_llm()
    if is_good:
        instruction = (
            "The gameplay is now complete. The character accumulated 2+ successes. "
            "Generate a GOOD ENDING — they somehow achieved their goal in the most accidental, "
            "embarrassing, or ridiculous way. Narrator reports it like bad news.\n"
            "Format:\n🎊 Good Ending: [Title In Title Case]\n"
            "[3-4 sentences of absurd accidental victory]\n"
            "[One ironic observation. Deadpan.]\n"
            '🏆 Achievement Unlocked: "[Absurd Name]"'
        )
    else:
        instruction = (
            "The gameplay is now complete. The character failed to reach 2 successes. "
            "Generate a BAD ENDING — they never achieved their goal.\n"
            "Format:\n🎭 Bad Ending: [Title In Title Case]\n"
            "[3-4 sentences of final catastrophic fate]\n"
            "[One ironic lesson. Deadpan.]\n"
            '🏆 Achievement Unlocked: "[Absurd Name]"'
        )
    # Trim history: ambil system prompt + max 5 pesan terakhir saja
    # supaya tidak melebihi context window saat session panjang
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system  = [m for m in messages if m["role"] != "system"]
    trimmed = system_msgs + non_system[-5:]  # 5 pesan = ~2-3 turn terakhir
    trimmed.append({"role": "user", "content": instruction})

    out      = llm.create_chat_completion(
        messages=trimmed,
        max_tokens=280,
        temperature=0.8,
        top_p=0.9,
        repeat_penalty=1.15,
    )
    response = out['choices'][0]['message']['content'].strip()
    response = strip_markdown(response)
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": response})
    return response, messages


# ── TTS ───────────────────────────────────────────────
def extract_narration(text):
    keep = []
    for line in text.split('\n'):
        line = ''.join(
            c for c in line
            if unicodedata.category(c) != 'So'
            and not (0x1F000 <= ord(c) <= 0x1FFFF)
            and not (0x2600  <= ord(c) <= 0x27FF)
        ).strip()
        if not line: continue
        if re.match(r'^(result:|world state:|\[?fail\]?$|\[?success)', line, re.IGNORECASE): continue
        if re.match(r'^(bad ending|good ending|achievement unlocked)', line, re.IGNORECASE): continue
        if re.match(r'^[-\->→➡►]', line):
            line = re.sub(r'^[-\->→➡►\s]+', '', line).strip()
            if line: keep.append(line)
            continue
        keep.append(line)
    return ' '.join(keep).strip()

SAMPLE_RATE = 24000

@spaces.GPU(duration=120)
def synthesize_audio(text, voice=NARRATOR_VOICE):
    """Returns (sample_rate, audio_array) or None for Gradio Audio."""
    try:
        tts   = load_tts()
        clean = extract_narration(text)
        if not clean:
            return None, []
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        chunks        = []
        sentence_ends = []   # cumulative sample count setelah tiap kalimat
        total_samples = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            sent_chunks = []
            for chunk in tts(sentence, voice=voice, speed=0.85):
                audio = chunk[2] if isinstance(chunk, tuple) else chunk.audio
                sent_chunks.append(audio)
            if sent_chunks:
                cat = np.concatenate(sent_chunks)
                chunks.append(cat)
                total_samples += len(cat)
                sentence_ends.append((sentence, total_samples / SAMPLE_RATE))
        if not chunks:
            return None, []
        return (SAMPLE_RATE, np.concatenate(chunks)), sentence_ends
    except Exception as e:
        print(f"TTS error: {e}")
        return None, []


# ── HTML rendering ─────────────────────────────────────
def render_turn_html(response, turn_num, successes):
    pip_full  = "▓" * successes
    pip_empty = "░" * max(0, 2 - successes)
    header = (
        f'<div class="turn-header">'
        f'TURN {turn_num}/{MAX_TURNS} &nbsp;|&nbsp; '
        f'<span class="pips">{pip_full}{pip_empty}</span> {successes}/2'
        f'</div>'
    )
    body_lines = []
    for line in response.split('\n'):
        line = line.strip()
        if not line: continue
        if re.match(r'⚡\s*Result:\s*Fail', line, re.IGNORECASE):
            body_lines.append('<div class="result fail">⚡ FAIL</div>')
        elif re.match(r'⚡\s*Result:\s*Success', line, re.IGNORECASE):
            body_lines.append('<div class="result success">⚡ SUCCESS</div>')
        elif line.startswith('➡️') or line.startswith('->') or line.startswith('→'):
            ws = re.sub(r'^[➡→\->\s]+', '', line).strip()
            body_lines.append(f'<div class="world-state">→ {ws}</div>')
        elif line.lower().rstrip('.') in _KNOWN_REACTIONS:
            body_lines.append(f'<div class="reaction">{line}</div>')
        else:
            body_lines.append(f'<div class="narration">{line}</div>')
    return f'<div class="turn-block">{header}{"".join(body_lines)}</div>'


def render_ending_html(response):
    lines = []
    for line in response.split('\n'):
        line = line.strip()
        if not line: continue
        # Skip achievement line — redundant dengan ending title
        if re.search(r'achievement unlocked', line, re.IGNORECASE):
            continue
        if re.match(r'(🎭|🎊)', line) or re.search(r'ending:', line, re.IGNORECASE):
            lines.append(f'<div class="ending-title">{line}</div>')
        else:
            lines.append(f'<div class="ending-text">{line}</div>')
    return f'<div class="ending-block">{"".join(lines)}</div>'


# ── Gradio event handlers ──────────────────────────────
def begin_game(name, age, gender, habit, scenario, first_action):
    for field, label in [(name, "Name"), (scenario, "Starting Scenario"), (first_action, "First Action")]:
        if not field.strip():
            raise gr.Error(f"'{label}' is required.")

    session   = [{"role": "system", "content": SYSTEM_PROMPT}]
    first_msg = (
        f"NAME: {name.strip()}\nAGE: {age.strip()}\nGENDER: {gender.strip()}\n"
        f"HABIT: {habit.strip()}\nSTARTING CONDITION: {scenario.strip()}\n"
        f"FIRST ACTION: {first_action.strip()}"
    )
    original_goal = scenario.strip()

    response, session = generate_turn(
        session, first_msg, turn_number=1,
        max_turns=MAX_TURNS + 1, original_goal=original_goal
    )

    audio, _timing = synthesize_audio(response)
    successes = count_total_successes(session)
    html_acc  = render_turn_html(response, 1, successes)
    status    = f"Turn 1 / {MAX_TURNS} &nbsp;|&nbsp; Successes: {successes} / 2"

    return (
        session,                        # state: session
        2,                              # state: turn_num
        original_goal,                  # state: goal
        html_acc,                       # state: accumulated html
        f'<div class="scroll-area">{html_acc}</div>',  # display
        audio,                          # audio
        status,                         # status md
        gr.update(visible=False),       # hide setup
        gr.update(visible=True),        # show game
        # action_in sudah di-handle oleh _lock_submit (queue=False)
    )


def build_subtitle_html(html_acc, sentences_so_far, is_current_turn=True):
    """Wrap accumulated html + subtitle lines untuk streaming effect."""
    if not sentences_so_far:
        return f'<div class="scroll-area">{html_acc}</div>'
    subtitle_divs = ''.join(
        f'<div class="subtitle-line">{s}</div>' for s in sentences_so_far
    )
    cls = "subtitle-block active" if is_current_turn else "subtitle-block"
    return f'<div class="scroll-area">{html_acc}<div class="{cls}">{subtitle_divs}</div></div>'


def submit_action(action, session, turn_num, original_goal, html_acc):
    if not action.strip():
        raise gr.Error("Type an action first.")
    if turn_num > MAX_TURNS:
        raise gr.Error("Game over. Refresh to start a new run.")

    response, session = generate_turn(
        session, action.strip(),
        turn_number=turn_num, max_turns=MAX_TURNS + 1,
        original_goal=original_goal
    )
    successes   = count_total_successes(session)
    turn_html   = render_turn_html(response, turn_num, successes)
    is_final    = (turn_num == MAX_TURNS)

    # ── Generate audio untuk turn ini ──
    audio, timing = synthesize_audio(response)
    status_turn   = f"Turn {turn_num} / {MAX_TURNS} &nbsp;|&nbsp; Successes: {successes} / 2"

    if not is_final:
        # Normal turn: return 11 items (ending_st="", audio_ending_st=None)
        new_html_acc = html_acc + turn_html
        return (
            session,
            turn_num + 1,
            original_goal,
            new_html_acc,
            f'<div class="scroll-area">{new_html_acc}</div>',
            audio,
            status_turn,
            gr.update(value="", interactive=True),
            gr.update(interactive=True, value="→"),
            "",    # ending_st — kosong, bukan final turn
            None,  # audio_ending_st — kosong
        )

    # ── FINAL TURN: turn audio dulu, ending menyusul via state ──
    # Simpan ending di state supaya bisa di-trigger setelah audio turn selesai
    is_good       = successes >= 2
    ending, session = generate_ending(session, is_good=is_good, original_goal=original_goal)
    ending_html   = render_ending_html(ending)
    audio_ending, _ = synthesize_audio(ending)

    new_html_acc  = html_acc + turn_html  # ending belum ditampilkan
    status_final  = (
        f"GAME OVER &nbsp;|&nbsp; Successes: {successes} / 2 &nbsp;|&nbsp; "
        f"{'🎊 GOOD ENDING' if is_good else '🎭 BAD ENDING'}"
    )

    # Return turn audio + html tanpa ending dulu
    # UI akan trigger show_ending setelah audio turn selesai via .then()
    return (
        session,
        turn_num + 1,
        original_goal,
        new_html_acc,                                       # state: belum ada ending
        f'<div class="scroll-area">{new_html_acc}</div>',  # display: belum ada ending
        audio,                                              # audio TURN (bukan ending)
        status_final,
        gr.update(value="", interactive=False),
        gr.update(interactive=False, value="Game Over"),
        ending_html,       # output ke ending_st (state baru)
        audio_ending,      # output ke audio_ending_st (state baru)
    )


def show_ending(html_acc, ending_html, audio_ending):
    """Dipanggil setelah audio turn 5 selesai — tampilkan ending."""
    full_html = html_acc + ending_html
    return (
        full_html,                                          # update html_acc_st
        f'<div class="scroll-area">{full_html}</div>',     # update narrative_html
        audio_ending,                                       # play audio ending
        gr.update(interactive=True, value="🔄 Play Again"),  # submit_btn jadi restart
    )


def restart_game():
    """Reset semua state, kembali ke setup screen."""
    return (
        [],   # session_st
        2,    # turn_st
        "",   # goal_st
        "",   # html_acc_st
        '<div class="scroll-area"></div>',  # narrative_html
        None, # audio_out
        '<div class="status-bar">Turn 1 / 5</div>',  # status_md
        gr.update(visible=True),   # setup_section
        gr.update(visible=False),  # game_section
        "",   # ending_st
        None, # audio_ending_st
    )


# ── CSS ────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=Share+Tech+Mono&display=swap');
body, .gradio-container { background: #0c0c0c !important; }
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-family: 'Special Elite', serif;
    font-size: 2.4rem;
    color: #e8d5a3;
    letter-spacing: 0.12em;
    margin: 0;
}
.app-header p {
    font-family: 'Share Tech Mono', monospace;
    color: #666;
    font-size: 0.85rem;
    margin: 0.4rem 0 0;
}
.scroll-area {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 1rem;
    min-height: 200px;
    max-height: 480px !important;
    overflow-y: auto !important;
    display: block !important;
    font-family: 'Share Tech Mono', monospace;
}
.turn-block {
    margin-bottom: 1.4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1e1e1e;
}
.turn-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    letter-spacing: 0.15em;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
}
.pips { color: #e8d5a3; }
.narration {
    font-family: 'Special Elite', serif;
    color: #c8b99a;
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 0.2rem 0;
}
.reaction {
    font-family: 'Share Tech Mono', monospace;
    color: #888;
    font-size: 0.8rem;
    margin: 0.5rem 0;
    font-style: italic;
}
.result {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    font-weight: bold;
    margin: 0.4rem 0;
    letter-spacing: 0.08em;
}
.result.fail    { color: #cc4444; }
.result.success { color: #44aa66; }
.world-state {
    font-family: 'Share Tech Mono', monospace;
    color: #5a7a6a;
    font-size: 0.8rem;
    margin-top: 0.4rem;
    padding-left: 0.5rem;
    border-left: 2px solid #2a3a2a;
}
.ending-block {
    margin-top: 1rem;
    padding: 1rem;
    background: #0d0d0d;
    border: 1px solid #3a2a1a;
    border-radius: 4px;
}
.ending-title {
    font-family: 'Special Elite', serif;
    color: #e8d5a3;
    font-size: 1.1rem;
    margin-bottom: 0.6rem;
}
.ending-text {
    font-family: 'Special Elite', serif;
    color: #b09070;
    font-size: 0.9rem;
    line-height: 1.7;
    margin: 0.2rem 0;
}
.achievement {
    font-family: 'Share Tech Mono', monospace;
    color: #888855;
    font-size: 0.75rem;
    margin-top: 0.8rem;
}
label { color: #888 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.8rem !important; }
input, textarea { background: #141414 !important; color: #c8b99a !important; border-color: #2a2a2a !important; font-family: 'Share Tech Mono', monospace !important; }
.status-bar { font-family: 'Share Tech Mono', monospace; color: #666; font-size: 0.75rem; text-align: center; padding: 0.3rem; }
/* ── Subtitle streaming ── */
.subtitle-block {
    margin-top: 0.8rem;
    padding: 0.6rem 0.8rem;
    border-left: 2px solid #2a2a2a;
    opacity: 0.85;
}
.subtitle-block.active {
    border-left-color: #4a6a8a;
    opacity: 1;
}
.subtitle-line {
    font-family: 'Special Elite', serif;
    color: #c8b99a;
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 0.15rem 0;
    animation: fadein 0.4s ease-in;
}
@keyframes fadein {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
/* ── Hide audio player controls, keep waveform ── */
.audio-player button,
.audio-player .icon-button,
audio::-webkit-media-controls-panel,
audio::-webkit-media-controls-play-button,
audio::-webkit-media-controls-timeline,
audio::-webkit-media-controls-current-time-display,
audio::-webkit-media-controls-time-remaining-display,
audio::-webkit-media-controls-mute-button,
audio::-webkit-media-controls-volume-slider {
    display: none !important;
}
/* Sembunyikan seluruh audio widget Gradio, audio tetap autoplay via JS */
.gradio-audio { display: none !important; }
/* ── Download button ── */
.download-btn {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    text-decoration: none;
    border: 1px solid #2a2a2a;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    display: inline-block;
    margin-top: 0.5rem;
    transition: color 0.2s, border-color 0.2s;
}
.download-btn:hover { color: #999; border-color: #444; }
"""

# ── Download helper ───────────────────────────────────
def build_download_html(html_acc, status):
    """Buat HTML file lengkap untuk download adventure."""
    import base64, datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Bad Ending Generator — Adventure Log</title>
<link href="https://fonts.googleapis.com/css2?family=Special+Elite&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
body {{ background: #0c0c0c; color: #c8b99a; font-family: 'Special Elite', serif;
       max-width: 700px; margin: 2rem auto; padding: 1rem; }}
h1 {{ font-size: 1.8rem; color: #e8d5a3; letter-spacing: 0.12em; text-align: center; }}
.meta {{ font-family: 'Share Tech Mono', monospace; color: #444; font-size: 0.75rem;
         text-align: center; margin-bottom: 2rem; }}
.turn-block {{ margin-bottom: 1.4rem; padding-bottom: 1rem;
               border-bottom: 1px solid #1e1e1e; }}
.turn-header {{ font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
                color: #444; letter-spacing: 0.15em; margin-bottom: 0.6rem; }}
.narration {{ line-height: 1.7; margin: 0.2rem 0; }}
.reaction {{ font-family: 'Share Tech Mono', monospace; color: #888;
             font-size: 0.8rem; font-style: italic; }}
.result {{ font-family: 'Share Tech Mono', monospace; font-size: 0.85rem;
           font-weight: bold; letter-spacing: 0.08em; }}
.result.fail {{ color: #cc4444; }} .result.success {{ color: #44aa66; }}
.world-state {{ font-family: 'Share Tech Mono', monospace; color: #5a7a6a;
                font-size: 0.8rem; padding-left: 0.5rem;
                border-left: 2px solid #2a3a2a; }}
.ending-block {{ padding: 1rem; background: #0d0d0d;
                 border: 1px solid #3a2a1a; border-radius: 4px; margin-top: 1rem; }}
.ending-title {{ font-size: 1.1rem; color: #e8d5a3; margin-bottom: 0.6rem; }}
.ending-text {{ color: #b09070; line-height: 1.7; }}
.pips {{ color: #e8d5a3; }}
</style>
</head>
<body>
<h1>BAD ENDING GENERATOR</h1>
<div class="meta">{status} &nbsp;·&nbsp; {timestamp}</div>
{html_acc}
</body>
</html>"""
    b64 = base64.b64encode(page.encode()).decode()
    return f'data:text/html;base64,{b64}'


# ── UI ─────────────────────────────────────────────────
with gr.Blocks(title="Bad Ending Generator") as demo:

    # State
    session_st      = gr.State([])
    turn_st         = gr.State(2)
    goal_st         = gr.State("")
    html_acc_st     = gr.State("")
    ending_st       = gr.State("")   # ending html, disimpan sampai audio turn 5 selesai
    audio_ending_st = gr.State(None) # audio ending, disimpan sampai audio turn 5 selesai
    is_final_st     = gr.State(False)
    status_st       = gr.State("")

    # Header
    # <h1>BAD ENDING GENERATOR</h1>
    # <p>An AI narrator watches your decisions with quiet disappointment.</p>
    gr.HTML("""
    <div class="app-header">
        <h1> </h1>
        <p> </p>
    </div>
    """)

    # ── Setup section ──────────────────────────────────
    with gr.Column(visible=True) as setup_section:
        gr.Markdown("### ")
        # gr.Markdown("### Who is our unfortunate protagonist?")
        with gr.Row():
            name_in   = gr.Textbox(label="Name",   placeholder="James",  scale=2)
            age_in    = gr.Textbox(label="Age",    placeholder="20",     scale=1)
            gender_in = gr.Textbox(label="Gender", placeholder="male",   scale=1)
        habit_in    = gr.Textbox(
            label="Habit",
            placeholder="laughing nervously when people look at him"
        )
        scenario_in = gr.Textbox(
            label="Starting Scenario",
            placeholder="Walking through the most crowded center atrium of a luxury shopping mall during rush hour. Suddenly, he trips over a flat floor transition and lets out a blood-curdling scream. Hundreds of shoppers stare.",
            lines=4
        )
        first_action_in = gr.Textbox(
            label="First Action",
            placeholder="pretend to faint"
        )
        begin_btn = gr.Button("BEGIN THE SUFFERING →", variant="primary", size="lg")

    # ── Game section (initially hidden) ───────────────
    with gr.Column(visible=False) as game_section:
        status_md      = gr.HTML('<div class="status-bar">Turn 1 / 5</div>')
        narrative_html = gr.HTML('<div class="scroll-area"></div>')
        # Audio hidden via CSS (.gradio-audio display:none), tetap autoplay
        audio_out      = gr.Audio(
            autoplay=True, show_label=False,
            streaming=False, interactive=False,
            elem_classes=["audio-player"],
        )
        with gr.Row():
            action_in  = gr.Textbox(
                placeholder="What does your character do?",
                show_label=False, scale=5, container=False
            )
            submit_btn = gr.Button(
                "→", variant="primary", scale=1, min_width=60,
                interactive=False,
            )
        # Download link — muncul setelah game over
        download_html = gr.HTML("", visible=False)

    # ── Output lists ───────────────────────────────────
    _begin_outputs = [
        session_st, turn_st, goal_st, html_acc_st,
        narrative_html, audio_out, status_md,
        setup_section, game_section,
    ]

    # submit_action bisa return 9 (normal) atau 11 (final turn) items
    # Gradio tidak support variadic outputs, jadi kita selalu return 11
    # dan tambah ending_st + audio_ending_st ke output list
    _submit_outputs = [
        session_st, turn_st, goal_st, html_acc_st,
        narrative_html, audio_out, status_md,
        action_in, submit_btn,
        ending_st, audio_ending_st,
    ]

    _show_ending_outputs = [
        html_acc_st, narrative_html, audio_out, submit_btn,
    ]

    _restart_outputs = [
        session_st, turn_st, goal_st, html_acc_st,
        narrative_html, audio_out, status_md,
        setup_section, game_section,
        ending_st, audio_ending_st,
    ]

    # ── Helper functions ───────────────────────────────
    def _lock_submit():
        return gr.update(interactive=False, value="→"), gr.update(value="")

    def _unlock_submit():
        return gr.update(interactive=True)

    def _pre_submit(action):
        if not action.strip():
            raise gr.Error("Ketik aksi dulu sebelum melanjutkan.")
        return gr.update(interactive=False, value="⏳")

    def _check_if_final_and_show_ending(ending_html, audio_ending, html_acc):
        """Dipanggil via .then() setelah audio turn 5 selesai — show ending."""
        if not ending_html:
            # Bukan turn final, tidak ada yang perlu ditampilkan
            return gr.update(), gr.update(), gr.update(), gr.update()
        full_html = html_acc + ending_html
        return (
            full_html,
            f'<div class="scroll-area">{full_html}</div>',
            audio_ending,
            gr.update(interactive=True, value="🔄 Play Again"),
        )

    def _handle_restart_or_submit(btn_value, action, session, turn_num, goal, html_acc):
        """submit_btn sekarang bisa jadi submit ATAU restart."""
        if btn_value == "🔄 Play Again":
            return restart_game()
        # else: normal submit — dipanggil dari _pre_submit chain

    def _make_download(html_acc, status):
        href = build_download_html(html_acc, status)
        return gr.update(
            visible=True,
            value=f'<a class="download-btn" href="{href}" download="bad-ending-adventure.html">⬇ Download Adventure Log</a>'
        )

    # ── Event wiring ───────────────────────────────────
    begin_btn.click(
        fn=_lock_submit,
        inputs=[],
        outputs=[submit_btn, action_in],
        api_name=False,
        queue=False,
    ).then(
        fn=begin_game,
        inputs=[name_in, age_in, gender_in, habit_in, scenario_in, first_action_in],
        outputs=_begin_outputs,
        api_name=False,
    ).then(
        fn=_unlock_submit,
        inputs=[],
        outputs=[submit_btn],
        api_name=False,
        queue=False,
    )

    submit_btn.click(
        fn=_pre_submit,
        inputs=[action_in],
        outputs=[submit_btn],
        api_name=False,
        queue=False,
        concurrency_limit=1,
    ).then(
        fn=submit_action,
        inputs=[action_in, session_st, turn_st, goal_st, html_acc_st],
        outputs=_submit_outputs,
        api_name=False,
        concurrency_limit=1,
    ).then(
        # Setelah audio (turn atau non-final) selesai:
        # cek apakah ada ending yang perlu ditampilkan
        fn=_check_if_final_and_show_ending,
        inputs=[ending_st, audio_ending_st, html_acc_st],
        outputs=_show_ending_outputs,
        api_name=False,
        queue=False,
    ).then(
        # Setelah ending tampil + audio ending selesai: tampilkan download link
        fn=_make_download,
        inputs=[html_acc_st, status_md],
        outputs=[download_html],
        api_name=False,
        queue=False,
    )

    # Play Again button (value berubah jadi "🔄 Play Again" saat game over)
    # Ini di-handle oleh submit_btn.click sendiri — tapi kita perlu tangkap
    # kasus dimana btn value = Play Again supaya tidak ke _pre_submit
    # Solusi: ganti fn _pre_submit untuk cek btn value via JS, atau
    # gunakan separate restart_btn yang di-show/hide
    # Approach lebih clean: separate invisible restart btn
    restart_btn = gr.Button("🔄 Play Again", visible=False, variant="secondary")

    def _swap_to_restart():
        """Ganti submit_btn dengan restart_btn saat game over."""
        return gr.update(visible=False), gr.update(visible=True)

    # Trigger swap setelah show_ending selesai — kita sudah handle di
    # submit_btn return value="🔄 Play Again" tapi itu membingungkan
    # Jadi: submit_btn tetap di-handle untuk submit, restart_btn untuk restart

    restart_btn.click(
        fn=restart_game,
        inputs=[],
        outputs=[
            session_st, turn_st, goal_st, html_acc_st,
            narrative_html, audio_out, status_md,
            setup_section, game_section,
            ending_st, audio_ending_st,
        ],
        api_name=False,
    ).then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=False, value="→")),
        inputs=[],
        outputs=[submit_btn, restart_btn, download_html, submit_btn],
        api_name=False,
        queue=False,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=CSS, max_threads=4)
