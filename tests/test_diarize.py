import unittest

from podtrans.diarize import (
    ROOT,
    Job,
    LabeledToken,
    TimedToken,
    assign_speakers,
    build_text_segments,
    cluster_backend_name,
    extract_timed_tokens,
    shard_jobs,
    snap_speaker_timeline,
    stabilize_speaker_units,
)


class ExtractTimedTokensTests(unittest.TestCase):
    def test_reconstructs_multichunk_transcript_exactly(self) -> None:
        payload = {
            "text": "你好。\n世界！",
            "result": {
                "chunks": [
                    {
                        "index": 0,
                        "start_seconds": 0,
                        "text": "你好。",
                        "result": {
                            "timestamps": [
                                {"token": "你", "start_time": 0.1, "end_time": 0.2},
                                {"token": "好", "start_time": 0.2, "end_time": 0.3},
                                {"token": "。", "start_time": 0.3, "end_time": 0.3},
                            ]
                        },
                    },
                    {
                        "index": 1,
                        "start_seconds": 30,
                        "text": "世界！",
                        "result": {
                            "timestamps": [
                                {"token": "世", "start_time": 0.1, "end_time": 0.2},
                                {"token": "界", "start_time": 0.2, "end_time": 0.3},
                                {"token": "！", "start_time": 0.3, "end_time": 0.3},
                            ]
                        },
                    },
                ]
            },
        }

        tokens = extract_timed_tokens(payload)

        self.assertEqual("".join(token.text for token in tokens), payload["text"])
        self.assertEqual(tokens[4].start_ms, 30100)
        self.assertTrue(tokens[3].separator)

    def test_interpolates_timestamp_text_mismatch_without_losing_text(self) -> None:
        payload = {
            "text": "你好",
            "result": {
                "chunks": [
                    {
                        "index": 0,
                        "start_seconds": 0,
                        "text": "你好",
                        "result": {
                            "timestamps": [
                                {"token": "你", "start_time": 0.1, "end_time": 0.2}
                            ]
                        },
                    }
                ]
            },
        }
        stats = {}

        tokens = extract_timed_tokens(payload, stats=stats)

        self.assertEqual("".join(token.text for token in tokens), "你好")
        self.assertEqual(stats["synthetic_timestamp_tokens"], 1)
        self.assertEqual(stats["chunks_with_timestamp_fallback"], 1)

    def test_interpolates_chunk_without_token_timestamps(self) -> None:
        payload = {
            "text": "无时间",
            "result": {
                "chunks": [
                    {
                        "index": 7,
                        "start_seconds": 10,
                        "end_seconds": 20,
                        "text": "无时间",
                        "result": {"text": "无时间"},
                    }
                ]
            },
        }
        stats = {}

        tokens = extract_timed_tokens(
            payload,
            audio_duration_seconds=30,
            stats=stats,
        )

        self.assertEqual("".join(token.text for token in tokens), payload["text"])
        self.assertEqual([token.start_ms for token in tokens], [10000, 13333, 16667])
        self.assertEqual([token.end_ms for token in tokens], [13333, 16667, 20000])
        self.assertEqual(stats["chunks_without_token_timestamps"], 1)
        self.assertEqual(stats["synthetic_timestamp_tokens"], 3)
        self.assertEqual(stats["chunks_with_timestamp_fallback"], 1)

    def test_supports_flat_short_audio_result(self) -> None:
        payload = {
            "text": "短通知。",
            "result": {
                "text": "短通知。",
                "timestamps": [
                    {"token": "短", "start_time": 0.1, "end_time": 0.2},
                    {"token": "通", "start_time": 0.2, "end_time": 0.3},
                    {"token": "知", "start_time": 0.3, "end_time": 0.4},
                    {"token": "。", "start_time": 0.4, "end_time": 0.4},
                ],
            },
        }
        stats = {}

        tokens = extract_timed_tokens(
            payload,
            audio_duration_seconds=1.0,
            stats=stats,
        )

        self.assertEqual("".join(token.text for token in tokens), payload["text"])
        self.assertEqual([token.start_ms for token in tokens], [100, 200, 300, 400])
        self.assertEqual(stats["flat_result_transcript"], 1)


class SpeakerAssignmentTests(unittest.TestCase):
    def test_assigns_by_maximum_timeline_overlap(self) -> None:
        tokens = [
            TimedToken("甲", 100, 300),
            TimedToken("乙", 900, 1100),
            TimedToken("丙", 1400, 1600),
        ]
        timeline = [[0.0, 1.0, 0], [1.0, 2.0, 1]]

        labeled = assign_speakers(tokens, timeline)

        self.assertEqual([token.speaker for token in labeled], [0, 0, 1])

    def test_snaps_change_away_from_middle_of_a_word(self) -> None:
        tokens = [
            TimedToken("聊", 0, 100),
            TimedToken("开", 120, 220),
            TimedToken("了", 240, 340),
            TimedToken("。", 360, 360),
            TimedToken("下", 700, 800),
        ]
        timeline = [[0.0, 0.25, 0], [0.25, 1.0, 1]]

        snapped, count = snap_speaker_timeline(tokens, timeline, max_shift_ms=500)

        self.assertEqual(count, 1)
        self.assertEqual(snapped[0][1], 0.53)


class TextSegmentTests(unittest.TestCase):
    def test_stabilizes_mid_sentence_speaker_jitter(self) -> None:
        tokens = [
            LabeledToken("聊", 0, 100, 0),
            LabeledToken("开", 120, 220, 0),
            LabeledToken("了", 240, 340, 1),
            LabeledToken("才", 360, 460, 2),
            LabeledToken("好", 480, 580, 0),
            LabeledToken("。", 600, 600, 0),
            LabeledToken("哈", 1000, 1100, 1),
            LabeledToken("！", 1120, 1120, 1),
        ]

        stabilized, changed = stabilize_speaker_units(tokens, min_gap_ms=250)

        self.assertEqual([token.speaker for token in stabilized[:6]], [0] * 6)
        self.assertEqual([token.speaker for token in stabilized[6:]], [1, 1])
        self.assertEqual(changed, 2)

    def test_segments_preserve_text_and_speaker_changes(self) -> None:
        tokens = [
            LabeledToken("你", 0, 100, 0),
            LabeledToken("好", 100, 200, 0),
            LabeledToken("。", 200, 200, 0),
            LabeledToken("\n", 300, 300, 0, separator=True),
            LabeledToken("世", 400, 500, 1),
            LabeledToken("界", 500, 600, 1),
            LabeledToken("！", 600, 600, 1),
        ]

        segments = build_text_segments(tokens, max_seconds=45, max_chars=400)

        self.assertEqual("".join(segment["text"] for segment in segments), "你好。\n世界！")
        self.assertEqual([segment["speaker_id"] for segment in segments], [0, 1])


class ClusterBackendTests(unittest.TestCase):
    def test_reports_funasr_backend_selected_by_window_count(self) -> None:
        self.assertIn("single-speaker", cluster_backend_name(19, None))
        self.assertIn("spectral", cluster_backend_name(2047, None))
        self.assertIn("UMAP", cluster_backend_name(2048, None))
        self.assertIn("spectral", cluster_backend_name(4096, 2))


class ShardingTests(unittest.TestCase):
    def test_path_hash_shards_are_stable_and_disjoint(self) -> None:
        jobs = []
        for index in range(40):
            episode_dir = ROOT / "test-channel" / f"{index:03d}-episode"
            output_json = episode_dir / "transcript_speakers.json"
            jobs.append(
                Job(
                    channel="test-channel",
                    number=str(index),
                    title=f"episode-{index}",
                    episode_dir=episode_dir,
                    audio_path=episode_dir / "audio.m4a",
                    transcript_json=episode_dir / "transcript.json",
                    output_json=output_json,
                    output_txt=output_json.with_suffix(".txt"),
                    output_md=output_json.with_suffix(".md"),
                )
            )

        memberships = [
            {
                job.audio_path
                for job in shard_jobs(jobs, shard_index=index, num_shards=4)
            }
            for index in range(4)
        ]

        self.assertEqual(set().union(*memberships), {job.audio_path for job in jobs})
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(memberships[left] & memberships[right])
        self.assertEqual(
            memberships,
            [
                {
                    job.audio_path
                    for job in shard_jobs(
                        list(reversed(jobs)),
                        shard_index=index,
                        num_shards=4,
                    )
                }
                for index in range(4)
            ],
        )


if __name__ == "__main__":
    unittest.main()
