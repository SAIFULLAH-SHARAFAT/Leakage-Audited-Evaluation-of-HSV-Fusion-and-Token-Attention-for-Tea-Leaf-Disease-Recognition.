import React, { useState } from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { PredictionResponse, XaiStats } from '@/types/api';
import { AGREEMENT_THRESHOLDS } from '@/constants/api';

// ── Single CAM overlay card ───────────────────────────────────────────────────
interface XAIOverlayProps {
    explanations: PredictionResponse['explanations'];
    method: keyof PredictionResponse['explanations'];
}

export function XAIOverlay({ explanations, method }: XAIOverlayProps) {
    let imageData = explanations[method];
    if (imageData && !imageData.startsWith('file://') && !imageData.startsWith('data:image')) {
        imageData = `data:image/png;base64,${imageData}`;
    }

    const label =
        method === 'consensus'    ? 'CONSENSUS (AVERAGED)'  :
        method === 'ablationcam'  ? 'ABLATION CAM'          :
        method.toUpperCase();

    return (
        <View style={styles.xaiCard}>
            <Text style={styles.methodTitle}>{label}</Text>
            {imageData ? (
                <Image source={{ uri: imageData }} style={styles.image} resizeMode="contain" />
            ) : (
                <View style={styles.placeholder}>
                    <Text style={styles.placeholderText}>No visualization available</Text>
                </View>
            )}
        </View>
    );
}

// ── Model-agreement reliability badge ─────────────────────────────────────────
interface ReliabilityBadgeProps { score: number; }

export function ReliabilityBadge({ score }: ReliabilityBadgeProps) {
    const level =
        score >= AGREEMENT_THRESHOLDS.STRONG       ? { label: 'High Agreement',     color: '#10B981' } :
        score >= AGREEMENT_THRESHOLDS.MODERATE_LOW ? { label: 'Moderate Agreement', color: '#F59E0B' } :
                                                     { label: 'Low Agreement',      color: '#DC2626' };
    return (
        <View style={[styles.badge, { backgroundColor: level.color + '20', borderColor: level.color }]}>
            <Text style={[styles.badgeText, { color: level.color }]}>
                {level.label}  (IoU {score.toFixed(2)})
            </Text>
        </View>
    );
}

// ── Quantitative XAI stats panel (new — addresses Reviewer #2) ────────────────
interface XAIStatsPanelProps { stats: XaiStats; }

export function XAIStatsPanel({ stats }: XAIStatsPanelProps) {
    const [expanded, setExpanded] = useState(false);

    const agreementColor = stats.agreement_pass ? '#10B981' : '#DC2626';
    const energyColor    = stats.energy_pass    ? '#10B981' : '#DC2626';

    return (
        <View style={styles.statsPanel}>
            {/* Header row */}
            <TouchableOpacity
                style={styles.statsPanelHeader}
                onPress={() => setExpanded(v => !v)}
                activeOpacity={0.7}
            >
                <Text style={styles.statsPanelTitle}>Explanation Quality Metrics</Text>
                <MaterialIcons
                    name={expanded ? 'expand-less' : 'expand-more'}
                    size={20}
                    color="#64748B"
                />
            </TouchableOpacity>

            {/* Summary row always visible */}
            <View style={styles.summaryRow}>
                <View style={styles.summaryCell}>
                    <Text style={styles.summaryCellLabel}>Inter-method Agreement</Text>
                    <Text style={[styles.summaryCellValue, { color: agreementColor }]}>
                        {stats.agreement_pass ? 'PASS' : 'FAIL'}
                        {'  '}(thr ≥ {stats.thresholds.agreement_iou.toFixed(2)})
                    </Text>
                </View>
                <View style={styles.summaryCell}>
                    <Text style={styles.summaryCellLabel}>Mean Energy Ratio</Text>
                    <Text style={[styles.summaryCellValue, { color: energyColor }]}>
                        {stats.mean_energy_ratio.toFixed(3)}
                        {'  '}(thr ≥ {stats.thresholds.energy_ratio.toFixed(2)})
                    </Text>
                </View>
            </View>

            {/* Per-method breakdown — collapsible */}
            {expanded && (
                <View style={styles.methodBreakdown}>
                    <Text style={styles.breakdownTitle}>Per-method Centre-Energy Ratio</Text>
                    {Object.entries(stats.per_method_energy).map(([method, ratio]) => {
                        const pass  = ratio >= stats.thresholds.energy_ratio;
                        const color = pass ? '#10B981' : '#DC2626';
                        const pct   = Math.round(ratio * 100);
                        return (
                            <View key={method} style={styles.breakdownRow}>
                                <Text style={styles.breakdownMethod}>{method.toUpperCase()}</Text>
                                <View style={styles.barTrack}>
                                    <View style={[styles.barFill, { width: `${pct}%`, backgroundColor: color }]} />
                                </View>
                                <Text style={[styles.breakdownValue, { color }]}>
                                    {ratio.toFixed(3)}
                                </Text>
                            </View>
                        );
                    })}
                    <Text style={styles.noteText}>
                        Centre-energy ratio = fraction of top-10 % active pixels inside
                        the central 60 %×60 % of the image.  Values below{' '}
                        {stats.thresholds.energy_ratio.toFixed(2)} indicate background bias.
                    </Text>
                </View>
            )}
        </View>
    );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
    xaiCard: {
        marginVertical: 10,
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 12,
    },
    methodTitle: {
        fontSize: 13,
        fontWeight: '600',
        color: '#0F172A',
        marginBottom: 8,
        letterSpacing: 0.4,
    },
    image: {
        width: '100%',
        height: 240,
        borderRadius: 8,
        backgroundColor: '#F1F5F9',
    },
    placeholder: {
        width: '100%',
        height: 240,
        borderRadius: 8,
        backgroundColor: '#F1F5F9',
        justifyContent: 'center',
        alignItems: 'center',
    },
    placeholderText: { color: '#94A3B8', fontSize: 14 },

    badge: {
        borderRadius: 8,
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderWidth: 1,
        alignSelf: 'flex-start',
        marginVertical: 10,
    },
    badgeText: { fontSize: 13, fontWeight: '600' },

    // ── stats panel ──────────────────────────────────────────────────────────
    statsPanel: {
        backgroundColor: '#F8FAFC',
        borderRadius: 12,
        borderWidth: 1,
        borderColor: '#E2E8F0',
        padding: 14,
        marginVertical: 10,
        gap: 10,
    },
    statsPanelHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    statsPanelTitle: {
        fontSize: 14,
        fontWeight: '600',
        color: '#0F172A',
    },
    summaryRow: {
        flexDirection: 'row',
        gap: 10,
    },
    summaryCell: {
        flex: 1,
        backgroundColor: '#FFFFFF',
        borderRadius: 8,
        padding: 10,
        borderWidth: 1,
        borderColor: '#E2E8F0',
    },
    summaryCellLabel: { fontSize: 11, color: '#64748B', marginBottom: 4 },
    summaryCellValue: { fontSize: 12, fontWeight: '700' },

    methodBreakdown: { gap: 8 },
    breakdownTitle: {
        fontSize: 12,
        fontWeight: '600',
        color: '#475569',
        marginBottom: 4,
    },
    breakdownRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    breakdownMethod: {
        width: 88,
        fontSize: 11,
        color: '#64748B',
        fontWeight: '600',
    },
    barTrack: {
        flex: 1,
        height: 8,
        backgroundColor: '#E2E8F0',
        borderRadius: 4,
        overflow: 'hidden',
    },
    barFill: { height: '100%', borderRadius: 4 },
    breakdownValue: { width: 46, fontSize: 11, fontWeight: '700', textAlign: 'right' },
    noteText: {
        fontSize: 11,
        color: '#94A3B8',
        lineHeight: 16,
        marginTop: 6,
        fontStyle: 'italic',
    },
});
