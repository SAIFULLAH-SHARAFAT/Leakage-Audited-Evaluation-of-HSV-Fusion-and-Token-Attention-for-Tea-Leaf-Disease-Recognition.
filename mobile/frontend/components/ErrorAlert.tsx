import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

interface ErrorAlertProps {
    error: string;
    onRetry?: () => void;
    onDismiss?: () => void;
}

export function ErrorAlert({ error, onRetry, onDismiss }: ErrorAlertProps) {
    return (
        <View style={styles.container}>
            <View style={styles.content}>
                <Text style={styles.title}>Error</Text>
                <Text style={styles.message}>{error}</Text>
                <View style={styles.buttonRow}>
                    {onDismiss && (
                        <TouchableOpacity style={[styles.button, styles.dismissButton]} onPress={onDismiss}>
                            <Text style={styles.dismissText}>Dismiss</Text>
                        </TouchableOpacity>
                    )}
                    {onRetry && (
                        <TouchableOpacity style={[styles.button, styles.retryButton]} onPress={onRetry}>
                            <Text style={styles.retryText}>Retry</Text>
                        </TouchableOpacity>
                    )}
                </View>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        padding: 16,
    },
    content: {
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 20,
        width: '100%',
        maxWidth: 400,
    },
    title: {
        fontSize: 18,
        fontWeight: '600',
        color: '#DC2626',
        marginBottom: 8,
    },
    message: {
        fontSize: 14,
        color: '#475569',
        marginBottom: 16,
        lineHeight: 20,
    },
    buttonRow: {
        flexDirection: 'row',
        gap: 12,
        justifyContent: 'flex-end',
    },
    button: {
        paddingVertical: 10,
        paddingHorizontal: 16,
        borderRadius: 8,
    },
    dismissButton: {
        backgroundColor: '#E2E8F0',
    },
    dismissText: {
        color: '#0F172A',
        fontWeight: '600',
    },
    retryButton: {
        backgroundColor: '#3B82F6',
    },
    retryText: {
        color: '#FFFFFF',
        fontWeight: '600',
    },
});
