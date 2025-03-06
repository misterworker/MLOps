'use client';

import { useState } from 'react';
import { 
  Container, 
  Title, 
  Text, 
  Button, 
  Group, 
  NumberInput, 
  Select, 
  SimpleGrid, 
  Paper, 
  Divider,
  Loader
} from '@mantine/core';
import axios from 'axios';
import brandModels from './brand_models';

export default function CarPricePredictor() {
  // State to store car details
  const [cars, setCars] = useState([
    {
      brand_model: '',
      location: '',
      year: undefined as number | undefined,
      kilometers_driven: undefined as number | undefined,
      fuel_type: '',
      transmission: '',
      owner_type: '',
      mileage: undefined as number | undefined,
      power: undefined as number | undefined,
      seats: undefined as number | undefined
    }
  ]);
  
  // State to store predicted prices after API response
  const [predictedPrices, setPredictedPrices] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);

  const locations = ["Ahmedabad", "Bangalore", "Chennai", "Coimbatore", "Delhi", "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune"];
  const fuelTypes = ["Petrol", "Diesel", "Other"];
  const transmissions = ["Manual", "Automatic"];
  const ownerTypes = ["First", "Second", "Third & Above"];

  // Utility functions to generate random values for examples
  const getRandomElement = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];
  const getRandomNumber = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;
  const getRandomFloat = (min: number, max: number, decimals: number) =>
    (Math.random() * (max - min) + min).toFixed(decimals);

  // Function to set an example car entry
  const setExample = (index: number) => {
    setCars((prevCars) => {
      const newCars = [...prevCars];
      newCars[index] = {
        brand_model: getRandomElement(brandModels),
        location: getRandomElement(locations),
        year: getRandomNumber(2000, 2025),
        kilometers_driven: getRandomNumber(10000, 200000),
        fuel_type: getRandomElement(fuelTypes),
        transmission: getRandomElement(transmissions),
        owner_type: getRandomElement(ownerTypes),
        mileage: parseFloat(getRandomFloat(10, 25, 1)),
        power: getRandomNumber(50, 300),
        seats: getRandomNumber(2, 7)
      };
      return newCars;
    });
  };

  // Function to update form values when user changes input
  const handleInputChange = (index: number, field: string, value: any) => {
    setCars((prevCars) => {
      const newCars = [...prevCars];
      newCars[index] = { ...newCars[index], [field]: value };
      return newCars;
    });
  };

  // Function to add a new car entry
  const addCar = () => {
    setCars([...cars, {
      brand_model: '',
      location: '',
      year: undefined,
      kilometers_driven: undefined,
      fuel_type: '',
      transmission: '',
      owner_type: '',
      mileage: undefined,
      power: undefined,
      seats: undefined
    }]);
  };

  // Function to remove a car entry
  const removeCar = (index: number) => {
    if (cars.length === 1) return; // Prevent removing the last input
    setCars(cars.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const cloudRunUrl = process.env.NEXT_PUBLIC_CLOUD_RUN_URL;

    if (!cloudRunUrl) {
      console.error('Cloud Run URL is not defined!');
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(cloudRunUrl, {
        "input": cars
      });

      if (Array.isArray(response.data)) {
        setPredictedPrices(response.data.map((car, index) =>
          car.predicted_price ? `Car ${index + 1}: ₹ ${car.predicted_price.toFixed(2)} Lakhs` : `Car ${index + 1}: Prediction Error`
        ));
      } else {
        setPredictedPrices(["Error in prediction."]);
      }
    } catch (error) {
      console.error("Prediction error", error);
      setPredictedPrices(["Error in prediction."]);
    } finally {
      setLoading(false); // after request completes
    }
  };

  return (
    <Container size="lg" py="xl">
      <Paper shadow="md" p="xl" radius="md" withBorder>
        <Title order={1} mb="md">Used Car Price Predictor</Title>
        <Text c="dimmed" mb="lg">Enter multiple car details to get price predictions.</Text>

        <form onSubmit={handleSubmit}>
          {cars.map((car, index) => (
            <Paper key={index} shadow="xs" p="md" withBorder mb="md">
              <Title order={3}>Car {index + 1}</Title>
              <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
                <Select
                  label="Brand & Model"
                  placeholder="Select brand & model"
                  data={brandModels}
                  value={car.brand_model}
                  onChange={(value) => handleInputChange(index, "brand_model", value || '')}
                  searchable
                  required
                />
                <Select
                  label="Location"
                  placeholder="Select location"
                  data={locations}
                  value={car.location}
                  onChange={(value) => handleInputChange(index, "location", value || '')}
                  required
                />
                <NumberInput
                  label="Year"
                  placeholder="2000-2025"
                  min={2000}
                  max={2025}
                  value={car.year}
                  onChange={(value) => handleInputChange(index, "year", typeof value === "number" ? value : undefined)}
                  required
                />
                <NumberInput
                  label="Kilometers Driven"
                  placeholder="e.g., 50000"
                  min={0}
                  value={car.kilometers_driven}
                  onChange={(value) => handleInputChange(index, "kilometers_driven", typeof value === "number" ? value : undefined)}
                  required
                />
                <Select
                  label="Fuel Type"
                  placeholder="Select fuel type"
                  data={fuelTypes}
                  value={car.fuel_type}
                  onChange={(value) => handleInputChange(index, "fuel_type", value || '')}
                  required
                />
                <Select
                  label="Transmission"
                  placeholder="Select transmission"
                  data={transmissions}
                  value={car.transmission}
                  onChange={(value) => handleInputChange(index, "transmission", value || '')}
                  required
                />
                <Select
                  label="Owner Type"
                  placeholder="Select owner type"
                  data={ownerTypes}
                  value={car.owner_type}
                  onChange={(value) => handleInputChange(index, "owner_type", value || '')}
                  required
                />
                <NumberInput
                  label="Mileage (km/l)"
                  placeholder="e.g., 18"
                  min={0}
                  value={car.mileage}
                  onChange={(value) => handleInputChange(index, "mileage", typeof value === "number" ? value : undefined)}
                  required
                />
                <NumberInput
                  label="Power (bhp)"
                  placeholder="e.g., 120"
                  min={0}
                  value={car.power}
                  onChange={(value) => handleInputChange(index, "power", typeof value === "number" ? value : undefined)}
                  required
                />
                <NumberInput
                  label="Seats"
                  placeholder="e.g., 5"
                  min={1}
                  max={10}
                  value={car.seats}
                  onChange={(value) => handleInputChange(index, "seats", typeof value === "number" ? value : undefined)}
                  required
                />
              </SimpleGrid>
              <Group mt="md">
                <Button color="gray" onClick={() => setExample(index)}>Set Example</Button>
                {cars.length > 1 && (
                  <Button color="red" onClick={() => removeCar(index)}>Remove</Button>
                )}
              </Group>
            </Paper>
          ))}

          <Group mt="md">
            <Button variant="outline" onClick={addCar}>+ Add Another Car</Button>
          </Group>

          <Group mt="xl">
            <Button type="submit" size="lg" variant="gradient" gradient={{ from: 'blue', to: 'cyan' }} disabled={loading}>
              {loading ? <Loader size="sm" color="white" /> : 'Predict Prices'}
            </Button>
          </Group>
        </form>
        {/* Display Predicted Prices Below Form(s) */}
        {predictedPrices !== null && (
          <>
            <Divider my="xl" />
            <Paper
              shadow="xs"
              p="md"
              withBorder
              style={{ textAlign: 'center', backgroundColor: '#f8f9fa' }}
            >
              {predictedPrices.map((price, index) => (
                <Text key={index} size="lg" c={price.includes("Error") ? "red" : "green"}>
                  {price}
                </Text>
              ))}
            </Paper>
          </>
        )}
      </Paper>
    </Container>
  );
}
