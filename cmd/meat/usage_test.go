package main

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestUsageFormatting(t *testing.T) {
	for i, line := range strings.Split(usage, "\n") {
		if utf8.RuneCountInString(line) > 80 {
			t.Errorf("line %d exceeds 80 columns: %q", i+1, line)
		}
		if strings.ContainsRune(line, '\t') || strings.TrimRight(line, " ") != line {
			t.Errorf("line %d contains tabs or trailing spaces: %q", i+1, line)
		}
	}

	for _, section := range []struct {
		heading string
		column  int
	}{
		{"Usage:", 25},
		{"Flags:", 25},
		{"Environment:", 25},
	} {
		t.Run(section.heading, func(t *testing.T) {
			_, body, ok := strings.Cut(usage, section.heading+"\n")
			if !ok {
				t.Fatal("missing section")
			}
			body, _, _ = strings.Cut(body, "\n\n")
			lines := strings.Split(body, "\n")
			for i, line := range lines {
				indent := len(line) - len(strings.TrimLeft(line, " "))
				if indent == 2 {
					// Long labels stay intact on their own line; the description
					// starts in the normal column on the following line.
					gap := strings.Index(line[2:], "  ")
					if gap < 0 {
						if len(line)+2 <= section.column {
							t.Errorf("short label unnecessarily occupies its own line: %q", line)
						}
						if i+1 == len(lines) || len(lines[i+1])-len(strings.TrimLeft(lines[i+1], " ")) != section.column {
							t.Errorf("long label must be followed by an aligned description: %q", line)
						}
						continue
					}
					start := 2 + gap
					indent = start + len(line[start:]) - len(strings.TrimLeft(line[start:], " "))
				}
				if indent != section.column {
					t.Errorf("description starts at column %d, want %d: %q", indent, section.column, line)
				}
			}
		})
	}
}
