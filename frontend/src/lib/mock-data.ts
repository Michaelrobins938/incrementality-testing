export interface Metric {
    label: string;
    value: string | number;
    change?: number;
    suffix?: string;
    status?: 'success' | 'warning' | 'danger' | 'neutral';
}

export interface ExperimentSummary {
    id: string;
    name: string;
    startDate: string;
    endDate: string;
    status: 'planning' | 'active' | 'analyzing' | 'complete';
    method: 'did' | 'synthetic_control' | 'bsts';
    geos: {
        treatment: number;
        control: number;
    };
    metrics: {
        incrementalLift: number;
        pValue: number;
        iroas: number;
        confidence: number;
    };
}

export interface TimeSeriesData {
    period: number;
    date: string;
    actual: number;
    counterfactual: number;
    upper?: number;
    lower?: number;
    isPostTreatment: boolean;
}

export const generateMockTimeSeries = (n_periods: number, treatment_start: number): TimeSeriesData[] => {
    const data: TimeSeriesData[] = [];
    const baseDate = new Date('2025-01-01');

    for (let i = 0; i < n_periods; i++) {
        const date = new Date(baseDate);
        date.setDate(date.getDate() + i * 7);

        const isPost = i >= treatment_start;
        const trend = 10000 + i * 200;
        const noise = (Math.random() - 0.5) * 1000;

        let actual = trend + noise;
        let counterfactual = trend + noise * 0.8;

        if (isPost) {
            const lift = 0.12; // 12% lift
            actual *= (1 + lift);
            // add a bit more noise to actual
            actual += (Math.random() - 0.5) * 500;
        }

        data.push({
            period: i,
            date: date.toISOString().split('T')[0],
            actual: Math.round(actual),
            counterfactual: Math.round(counterfactual),
            upper: Math.round(counterfactual * 1.05),
            lower: Math.round(counterfactual * 0.95),
            isPostTreatment: isPost
        });
    }

    return data;
};

export const MOCK_EXPERIMENT: ExperimentSummary = {
    id: 'exp-001',
    name: 'Q1 Facebook Performance Holdout',
    startDate: '2025-01-01',
    endDate: '2025-03-26',
    status: 'complete',
    method: 'did',
    geos: {
        treatment: 25,
        control: 25
    },
    metrics: {
        incrementalLift: 0.123,
        pValue: 0.0024,
        iroas: 3.45,
        confidence: 0.95
    }
};
