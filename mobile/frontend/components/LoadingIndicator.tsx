import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';

interface LoadingIndicatorProps {
    message?: string;
}

export function LoadingIndicator({ message = 'Loading...' }: LoadingIndicatorProps) {
    return (
        <View style={styles.container}>
            <ActivityIndicator size="large" color="#3B82F6" />
            <Text style={styles.text}>{message}</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#FFFFFF',
    },
    text: {
        marginTop: 12,
        fontSize: 16,
        color: '#64748B',
    },
});
