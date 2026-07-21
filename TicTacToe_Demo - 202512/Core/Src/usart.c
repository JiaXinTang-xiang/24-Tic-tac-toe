/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    usart.c
  * @brief   This file provides code for the configuration
  *          of the USART instances.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
	#include "StepMotor.h"
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "usart.h"

/* USER CODE BEGIN 0 */
uint8_t RecBuf[17]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};//���ڽ�������
uint8_t UART1_Rx_flg = 0;//���ڽ��������жϱ�־
uint16_t UART1_Rx_cnt = 17;//���ڽ������ݸ���



uint8_t RecBuf2[12]={0,1,2,3,4,5,7,8,9,10,11};//���12���ֽڵ�λ������	
uint16_t PosBuf0[3] = {0,0,0};//ȡ���ӵ�Դλ������
uint16_t PosBuf1[3] = {0,0,0};//�����ӵ�Ŀ��λ������

/* USER CODE END 0 */

UART_HandleTypeDef huart1;

/* USART1 init function */

void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */
		HAL_UART_Receive_IT(&huart1, RecBuf, 17);
  /* USER CODE END USART1_Init 2 */

}

void HAL_UART_MspInit(UART_HandleTypeDef* uartHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(uartHandle->Instance==USART1)
  {
  /* USER CODE BEGIN USART1_MspInit 0 */

  /* USER CODE END USART1_MspInit 0 */
    /* USART1 clock enable */
    __HAL_RCC_USART1_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**USART1 GPIO Configuration
    PA9     ------> USART1_TX
    PA10     ------> USART1_RX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* USART1 interrupt Init */
    HAL_NVIC_SetPriority(USART1_IRQn, 2, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
  /* USER CODE BEGIN USART1_MspInit 1 */

  /* USER CODE END USART1_MspInit 1 */
  }
}

void HAL_UART_MspDeInit(UART_HandleTypeDef* uartHandle)
{

  if(uartHandle->Instance==USART1)
  {
  /* USER CODE BEGIN USART1_MspDeInit 0 */

  /* USER CODE END USART1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_USART1_CLK_DISABLE();

    /**USART1 GPIO Configuration
    PA9     ------> USART1_TX
    PA10     ------> USART1_RX
    */
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_9|GPIO_PIN_10);

    /* USART1 interrupt Deinit */
    HAL_NVIC_DisableIRQ(USART1_IRQn);
  /* USER CODE BEGIN USART1_MspDeInit 1 */

  /* USER CODE END USART1_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */
uint8_t USART1_RecCommand(void)//���ڽ�������
{
	uint8_t temp = 0;

////		if(HAL_UART_Receive(&huart1,RecBuf,17,100) == HAL_OK)//���յ�17���ֽ�����
		
		//�������ݣ���Ҫ�ǵ����ڶ����ֽ�У��ֵ,��X��Y���겻�ܳ�������������˶��ķ�Χ����
//AA 0F A1 00 C8 00 00 00 00 04 00 00 00 00 00 62 55	//��ȷ����������1
//AA 0F A1 00 00 00 64 00 00 00 00 03 80 00 00 49 55	//��ȷ����������2
//AA 0F A1 00 96 00 96 00 00 04 01 03 16 00 00 BE 55 //��ȷ����������3

//AA 0F A1 01 12 01 AB 00 00 02 34 02 45 00 00 68 55//����,У��λ����68��Ӧ��Ϊ66
//AA 0F A1 09 12 01 AB 00 00 02 34 02 45 00 00 6E 55//����,��һ��X���곬����
//AA 0F A1 01 12 01 AB 00 00 02 34 09 45 00 00 6D 55//����,�ڶ���Y���곬����

	
	
//			HAL_UART_Transmit(&huart1,RecBuf,17,10); //��ʾ���У��ֵ ,����ʱ��������

			temp = RecBuf[1];			
			for(uint8_t i = 2;i<=14;i++)//��У��ֵ
      {
				temp = temp ^ RecBuf[i];
      }	
			
//			HAL_UART_Transmit(&huart1,&temp,1,10); //��ʾ���У��ֵ ,����ʱ��������
			
			if(RecBuf[0] == 0xAA && RecBuf[16] == 0x55 && RecBuf[15] == temp)
			{				

				PosBuf0[0] = RecBuf[3]<<8 | RecBuf[4];//3��16λ����,Դλ������
				PosBuf0[1] = RecBuf[5]<<8 | RecBuf[6];
				PosBuf0[2] = RecBuf[7]<<8 | RecBuf[8];
				PosBuf1[0] = RecBuf[9]<<8 | RecBuf[10];//3��16λ���ݣ�Ŀ��λ������
				PosBuf1[1] = RecBuf[11]<<8 | RecBuf[12];
				PosBuf1[2] = RecBuf[13]<<8 | RecBuf[14];
				if(PosBuf0[0] <= MAXPosX && PosBuf0[1] <= MAXPosY && PosBuf1[0] <= MAXPosX && PosBuf1[1] <= MAXPosY)//�ж����귶Χ
				{
					//��6��16λ����ת��Ϊ12���ֽڣ����ڷ�����֤
					RecBuf2[0] = PosBuf0[0]>>8;//ȡ��8λ
					RecBuf2[1] = PosBuf0[0];//ȡ��8λ
					RecBuf2[2] = PosBuf0[1]>>8;
					RecBuf2[3] = PosBuf0[1];
					RecBuf2[4] = PosBuf0[2]>>8;
					RecBuf2[5] = PosBuf0[2];
					RecBuf2[6] = PosBuf1[0]>>8;//ȡ��8λ
					RecBuf2[7] = PosBuf1[0];//ȡ��8λ
					RecBuf2[8] = PosBuf1[1]>>8;
					RecBuf2[9] = PosBuf1[1];
					RecBuf2[10] = PosBuf1[2]>>8;
					RecBuf2[11] = PosBuf1[2];				
//					HAL_UART_Transmit(&huart1,RecBuf2,12,10); //��ʾ��ȡ������ֵ������ʱ��������	
					return(1);
				}
			}
			

			Beep(200);//�������������������������⣬ԭ��Ϊ��У��ֵ���Ի�X��Y���곬������������˶��ķ�Χ
			return(0);

	
}

/* USER CODE END 1 */
