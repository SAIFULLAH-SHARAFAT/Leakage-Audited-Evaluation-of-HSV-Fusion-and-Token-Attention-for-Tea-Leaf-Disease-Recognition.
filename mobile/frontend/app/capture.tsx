import React, { useCallback, useEffect, useState } from 'react';
import {
    View,
    ScrollView,
    StyleSheet,
    Image,
    Text,
    TouchableOpacity,
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { usePredict } from '@/hooks/usePredict';
import { LoadingIndicator } from '@/components/LoadingIndicator';
import { ErrorAlert } from '@/components/ErrorAlert';
import { ActionButton } from '@/components/ActionButton';
import { predictionStore } from '@/constants/store';

export default function CaptureScreen() {
    const router = useRouter();
    const { predict, loading, error, data, reset } = usePredict();
    const [selectedImage, setSelectedImage] = useState<string | null>(null);

    const requestPermission = useCallback(async () => {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== ImagePicker.PermissionStatus.GRANTED) {
            Alert.alert(
                'Permission Required',
                'Please allow access to your photo library to select an image.'
            );
            return false;
        }
        return true;
    }, []);

    const requestCameraPermission = useCallback(async () => {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== ImagePicker.PermissionStatus.GRANTED) {
            Alert.alert(
                'Permission Required',
                'Please allow camera access to take a photo.'
            );
            return false;
        }
        return true;
    }, []);

    const pickImageFromLibrary = useCallback(async () => {
        const allowed = await requestPermission();
        if (!allowed) return;

        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Images,
                allowsEditing: true,
                aspect: [1, 1],
                quality: 1,
            });

            if (!result.canceled) {
                setSelectedImage(result.assets[0].uri);
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to pick image from library';
            console.error('Image picker error:', err);
            Alert.alert('Error', message);
        }
    }, [requestPermission]);

    const takePhoto = useCallback(async () => {
        const allowed = await requestCameraPermission();
        if (!allowed) return;

        try {
            const result = await ImagePicker.launchCameraAsync({
                allowsEditing: true,
                aspect: [1, 1],
                quality: 1,
            });

            if (!result.canceled) {
                setSelectedImage(result.assets[0].uri);
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to take photo';
            console.error('Camera error:', err);
            Alert.alert('Error', message);
        }
    }, [requestCameraPermission]);

    const handleAnalyze = useCallback(async () => {
        if (!selectedImage) {
            Alert.alert('No Image', 'Please select an image first');
            return;
        }

        await predict(selectedImage);
    }, [selectedImage, predict]);

    const handleRetry = useCallback(() => {
        reset();
        setSelectedImage(null);
    }, [reset]);

    useEffect(() => {
        if (!data || loading) return;

        predictionStore.setPrediction(data, selectedImage);

        router.replace({
            pathname: '/results',
            params: {
                status: data.final_status,
            },
        });
    }, [data, loading, router, selectedImage]);

    if (loading) {
        return <LoadingIndicator message="Analyzing leaf image..." />;
    }

    if (error) {
        return (
            <ErrorAlert
                error={error}
                onRetry={handleRetry}
                onDismiss={handleRetry}
            />
        );
    }

    return (
        <SafeAreaView style={styles.safeArea}>
            <ScrollView contentContainerStyle={styles.container}>
                <View style={styles.header}>
                    <Text style={styles.title}>Leaf Disease Checker</Text>
                    <Text style={styles.subtitle}>Take or upload a clear photo of the leaf</Text>
                </View>

                <View style={styles.imageCard}>
                    <Text style={styles.cardLabel}>Select Image</Text>
                    {selectedImage ? (
                        <Image source={{ uri: selectedImage }} style={styles.preview} />
                    ) : (
                        <View style={styles.placeholderBox}>
                            <MaterialIcons name="image" size={48} color="#94A3B8" />
                            <Text style={styles.placeholderText}>No image selected</Text>
                        </View>
                    )}

                    <View style={styles.buttonRow}>
                        <TouchableOpacity
                            style={styles.pickButton}
                            onPress={takePhoto}
                            disabled={loading}
                        >
                            <MaterialIcons name="camera-alt" size={20} color="#FFFFFF" />
                            <Text style={styles.pickButtonText}>Take Photo</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.pickButton}
                            onPress={pickImageFromLibrary}
                            disabled={loading}
                        >
                            <MaterialIcons name="photo-library" size={20} color="#FFFFFF" />
                            <Text style={styles.pickButtonText}>Choose Library</Text>
                        </TouchableOpacity>
                    </View>
                </View>

                {selectedImage && (
                    <View style={styles.actionCard}>
                        <Text style={styles.cardLabel}>Ready to Analyze</Text>
                        <Text style={styles.helperText}>
                            This will send the image to the backend for disease classification and
                            explanation analysis.
                        </Text>

                        <View style={styles.buttonRow}>
                            <ActionButton
                                icon="close"
                                label="Clear"
                                onPress={handleRetry}
                                variant="secondary"
                            />
                            <ActionButton
                                icon="check-circle"
                                label="Analyze"
                                onPress={handleAnalyze}
                                disabled={loading}
                            />
                        </View>
                    </View>
                )}

                <View style={styles.infoBox}>
                    <MaterialIcons name="info" size={20} color="#3B82F6" />
                    <Text style={styles.infoText}>
                        For best results, ensure the leaf is clearly visible, well-lit, and
                        centered in the frame.
                    </Text>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safeArea: {
        flex: 1,
        backgroundColor: '#F8FAFC',
    },
    container: {
        padding: 16,
        gap: 20,
    },
    header: {
        marginBottom: 8,
    },
    title: {
        fontSize: 26,
        fontWeight: '700',
        color: '#0F172A',
        marginBottom: 4,
    },
    subtitle: {
        fontSize: 14,
        color: '#64748B',
    },
    imageCard: {
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 16,
        gap: 12,
    },
    cardLabel: {
        fontSize: 16,
        fontWeight: '600',
        color: '#0F172A',
    },
    preview: {
        width: '100%',
        height: 280,
        borderRadius: 10,
        backgroundColor: '#E2E8F0',
    },
    placeholderBox: {
        width: '100%',
        height: 280,
        borderRadius: 10,
        backgroundColor: '#F1F5F9',
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 2,
        borderColor: '#E2E8F0',
        borderStyle: 'dashed',
    },
    placeholderText: {
        color: '#94A3B8',
        marginTop: 8,
        fontSize: 14,
    },
    pickButton: {
        flex: 1,
        flexDirection: 'row',
        backgroundColor: '#3B82F6',
        borderRadius: 10,
        paddingVertical: 12,
        paddingHorizontal: 16,
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
    },
    pickButtonText: {
        color: '#FFFFFF',
        fontWeight: '600',
        fontSize: 14,
    },
    actionCard: {
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 16,
        gap: 12,
    },
    helperText: {
        fontSize: 13,
        color: '#64748B',
        lineHeight: 18,
    },
    buttonRow: {
        flexDirection: 'row',
        gap: 12,
    },
    infoBox: {
        flexDirection: 'row',
        backgroundColor: '#EFF6FF',
        borderRadius: 10,
        padding: 12,
        gap: 12,
    },
    infoText: {
        flex: 1,
        fontSize: 13,
        color: '#1E40AF',
        lineHeight: 18,
    },
});
