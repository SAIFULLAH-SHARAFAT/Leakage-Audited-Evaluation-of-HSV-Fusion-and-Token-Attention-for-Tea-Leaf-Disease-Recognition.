import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';

interface ActionButtonProps {
    icon: string;
    label: string;
    onPress: () => void;
    variant?: 'primary' | 'secondary' | 'danger';
    disabled?: boolean;
}

export function ActionButton({
    icon,
    label,
    onPress,
    variant = 'primary',
    disabled = false,
}: ActionButtonProps) {
    return (
        <TouchableOpacity
            style={[styles.button, styles[variant], disabled && styles.disabled]}
            onPress={onPress}
            disabled={disabled}
        >
            <MaterialIcons
                name={icon as any}
                size={24}
                color={variant === 'secondary' ? '#0F172A' : '#FFFFFF'}
                style={styles.icon}
            />
            <Text style={[styles.label, variant === 'secondary' && styles.secondaryLabel]}>
                {label}
            </Text>
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
    button: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 12,
        paddingHorizontal: 16,
        borderRadius: 10,
        gap: 8,
    },
    primary: {
        backgroundColor: '#3B82F6',
    },
    secondary: {
        backgroundColor: '#E2E8F0',
    },
    danger: {
        backgroundColor: '#DC2626',
    },
    icon: {
        marginRight: 4,
    },
    label: {
        fontSize: 14,
        fontWeight: '600',
        color: '#FFFFFF',
    },
    secondaryLabel: {
        color: '#0F172A',
    },
    disabled: {
        opacity: 0.5,
    },
});
