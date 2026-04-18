# HoloAssist Task Procedures

This directory contains standardized procedure (SOP) JSON files for each task type in the HoloAssist evaluation dataset. These procedures describe the expected steps for tasks WITHOUT timestamps - candidates must detect when steps happen during video evaluation.

## Generated Procedures

### 1. **change_circuit_breaker.json**
- **Task Name**: Change Circuit Breaker
- **Description**: Replace a damaged circuit breaker in an electrical panel following proper safety protocols
- **Steps**: 11 canonical steps
- **Clips Analyzed**: 6 clips
- **Step Count Range**: 11-15 steps across source videos
- **Safety Focus**: Electrical safety, power shutdown, proper installation techniques

### 2. **assemble_computer_ram.json**
- **Task Name**: Install RAM Module
- **Description**: Install a RAM module into a computer motherboard correctly
- **Steps**: 14 canonical steps
- **Clips Analyzed**: 2 clips
- **Step Count Range**: 13-14 steps across source videos
- **Safety Focus**: ESD prevention, proper seating, power management

### 3. **assemble_computer_graphics_card.json**
- **Task Name**: Install Graphics Card
- **Description**: Install a graphics card into a computer motherboard and secure it properly
- **Steps**: 12 canonical steps
- **Clips Analyzed**: 2 clips
- **Step Count Range**: 11-12 steps across source videos
- **Safety Focus**: ESD prevention, proper slot management, securing bracket

### 4. **fix_motorcycle.json**
- **Task Name**: Fix Motorcycle (ATV)
- **Description**: Perform maintenance and repairs on a motorcycle or ATV following proper procedures
- **Steps**: 12 canonical steps
- **Clips Analyzed**: 7 clips
- **Step Count Range**: 9-13 steps across source videos
- **Safety Focus**: Vehicle support, power disconnection, tool safety

### 5. **setup_camera_dslr.json**
- **Task Name**: Setup DSLR Camera
- **Description**: Prepare a DSLR camera for use including battery, memory card, and basic configuration
- **Steps**: 8 canonical steps
- **Clips Analyzed**: 13 clips
- **Step Count Range**: 8-10 steps across source videos
- **Safety Focus**: Sensor protection, battery/card handling, lens care

### 6. **setup_gopro.json**
- **Task Name**: Setup GoPro Camera
- **Description**: Prepare a GoPro action camera for use including battery, memory card, and protective housing
- **Steps**: 11 canonical steps
- **Clips Analyzed**: 20 clips
- **Step Count Range**: 2-15 steps across source videos (widest variation)
- **Safety Focus**: Housing integrity, battery polarity, memory card security

## File Format

Each procedure JSON file follows this structure:

```json
{
  "task_id": "string",                    // Unique task identifier
  "task_name": "string",                  // Human-readable task name
  "description": "string",                // Detailed description
  "safety_notes": [                       // Array of safety considerations
    "string",
    ...
  ],
  "steps": [
    {
      "step_id": number,                  // Sequential step number
      "description": "string",            // What the step involves
      "expected_actions": [               // Verbs/nouns that define the action
        "string",                         // e.g., "turn_on", "insert", "cable"
        ...
      ],
      "common_errors": []                 // Placeholder for error patterns
    },
    ...
  ],
  "metadata": {
    "clips_analyzed": number,             // How many clips were analyzed
    "step_count_range": {
      "min": number,                      // Minimum steps in source clips
      "max": number,                      // Maximum steps in source clips
      "canonical": number                 // Steps in canonical procedure
    }
  }
}
```

## Methodology

### Step Identification
- Extracted coarse-grained actions from the HoloAssist annotation dataset
- Identified canonical procedures by selecting the median-complexity clip from each task type
- Deduplicated similar/redundant actions across multiple performers

### Safety Notes
- Compiled domain-specific safety protocols for each task
- Focused on critical safety concerns and best practices
- Included guidance on tool use, personal safety, and equipment protection

### Expected Actions
- Mapped action verbs (turn_on, insert, withdraw, etc.) from annotation data
- Extracted nouns representing objects (battery, cable, lens, etc.)
- Created expected_actions list for candidate evaluation: pure verb, pure noun, and combined phrase

### Variation Handling
- Different performers vary in step count and order
- Canonical procedures represent representative workflows, not prescriptive sequences
- Metadata captures the range of step counts across source videos to show acceptable variation

## Usage in Evaluation

Candidates receive these procedures and must:
1. Watch a video of someone performing the task
2. Identify when each procedure step occurs
3. Detect deviations from the canonical procedure
4. Flag safety violations or incorrect step ordering

The procedures serve as ground truth for temporal localization and step completion evaluation.

## Data Sources

- **Generation Script**: `generate_procedures.py`

## Summary Statistics

| Task | Steps | Clips | Step Range |
|------|-------|-------|------------|
| Change Circuit Breaker | 11 | 6 | 11-15 |
| Install RAM | 14 | 2 | 13-14 |
| Install Graphics Card | 12 | 2 | 11-12 |
| Fix Motorcycle | 12 | 7 | 9-13 |
| Setup DSLR Camera | 8 | 13 | 8-10 |
| Setup GoPro | 11 | 20 | 2-15 |

**Total**: 6 task types, 50 video clips, 6 procedure files
